"""FFmpeg renderer: crop, scale, subtitle burn-in, audio normalization."""

import json
import shutil
import subprocess
from pathlib import Path
from video_maker.config import FFMPEG_DIR, SUBTITLE_WORDS_PER_CHUNK, SUBTITLE_UPPERCASE
from video_maker.models import ClipSegment, FaceKeyframe, RenderPreset
from video_maker.utils import logger, words_to_hormozi_ass, clamp

import sys as _sys
FFMPEG_BIN = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))
FFPROBE_BIN = str(FFMPEG_DIR / ("ffprobe.exe" if _sys.platform == "win32" else "ffprobe"))

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


# ── Helpers ─────────────────────────────────────────────────────────


def _probe_video(source_path: Path) -> dict:
    """Get real video dimensions and duration via ffprobe."""
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(source_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.warning(f"ffprobe failed: {result.stderr[:300]}")
        return {"width": 1920, "height": 1080}

    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    return {
        "width": int(stream.get("width", 1920)),
        "height": int(stream.get("height", 1080)),
    }


def _average_face_x(keyframes: list[FaceKeyframe], fallback: int) -> int:
    """Compute the average face X across all keyframes."""
    if not keyframes:
        return fallback
    return int(sum(kf.x for kf in keyframes) / len(keyframes))


# Maximum keyframes in the FFmpeg expression (avoid overly deep nesting)
_MAX_EXPR_KEYFRAMES = 60


def _build_dynamic_crop_x(
    keyframes: list[FaceKeyframe],
    crop_w: int,
    src_width: int,
    clip_duration: float,
) -> str | None:
    """Build an FFmpeg expression for time-varying crop X with smooth face tracking.

    Generates a piecewise-linear interpolation between face positions,
    smoothed with a moving average for fluid motion.

    Returns an FFmpeg expression string (commas escaped for filter chain),
    or None if not enough keyframes.
    """
    if len(keyframes) < 2:
        return None

    half = crop_w // 2
    max_x = max(0, src_width - crop_w)

    # Convert face center X → crop_x (left edge of crop window), clamped
    # Keyframes arrive already EMA-smoothed and dead-zone filtered from vision.py
    times = [kf.time for kf in keyframes]
    crop_xs = [int(clamp(kf.x - half, 0, max_x)) for kf in keyframes]

    # Subsample if too many keyframes (keep expression depth manageable)
    if len(times) > _MAX_EXPR_KEYFRAMES:
        step = len(times) / _MAX_EXPR_KEYFRAMES
        indices = [int(i * step) for i in range(_MAX_EXPR_KEYFRAMES - 1)] + [len(times) - 1]
        times = [times[i] for i in indices]
        crop_xs = [crop_xs[i] for i in indices]

    # Build nested if(lt(t, t_i), smoothstep_interp, fallback) expression
    # Smoothstep: p = (t-t0)/dt; eased = p*p*(3-2*p); x = x0 + (x1-x0)*eased
    # Gives natural ease-in / ease-out camera movement
    expr = str(crop_xs[-1])  # last position as ultimate fallback

    for i in range(len(times) - 2, -1, -1):
        t0, t1 = times[i], times[i + 1]
        x0, x1 = crop_xs[i], crop_xs[i + 1]
        dt = t1 - t0
        if dt < 0.01:
            continue

        if x0 == x1:
            seg = str(x0)
        else:
            # Smoothstep easing: p*(p*(3-2*p)) where p = (t-t0)/dt
            p = f"(t-{t0:.2f})/{dt:.2f}"
            seg = f"{x0}+{x1 - x0}*{p}*{p}*(3-2*{p})"

        expr = f"if(lt(t,{t1:.2f}),{seg},{expr})"

    # Clamp to valid range
    expr = f"clip({expr},0,{max_x})"

    # Escape commas so FFmpeg filter chain doesn't split on them
    return expr.replace(",", "\\,")


def _compute_crop_window(real_w: int, real_h: int) -> tuple[int, int, int]:
    """Compute the 9:16 crop window dimensions and y offset for a landscape source.

    Returns (crop_w, crop_h, crop_y).
    """
    crop_h = real_h
    crop_w = int(real_h * 9 / 16)
    if crop_w > real_w:
        crop_w = real_w
        crop_h = int(real_w * 16 / 9)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    crop_y = int(clamp((real_h - crop_h) // 2, 0, real_h - crop_h))
    return crop_w, crop_h, crop_y


def _build_crop_scale_filters(
    segment: ClipSegment,
    real_w: int,
    real_h: int,
    duration: float,
    log_prefix: str,
) -> list[str]:
    """Build the crop + scale filter chain (without subtitles or pixel format)."""
    crop_w, crop_h, crop_y = _compute_crop_window(real_w, real_h)
    filters = []

    crop_x_expr = _build_dynamic_crop_x(segment.face_keyframes, crop_w, real_w, duration)
    if crop_x_expr:
        filters.append(f"crop={crop_w}:{crop_h}:{crop_x_expr}:{crop_y}")
        logger.info(
            f"{log_prefix}Dynamic crop: {crop_w}x{crop_h}, "
            f"{len(segment.face_keyframes)} keyframes, y={crop_y}"
        )
    else:
        face_x = _average_face_x(segment.face_keyframes, real_w // 2)
        if not segment.face_keyframes:
            logger.warning(f"{log_prefix}No face detected — centering crop")
        crop_x = int(clamp(face_x - crop_w // 2, 0, real_w - crop_w))
        filters.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
        logger.info(f"{log_prefix}Static crop: {crop_w}x{crop_h} at ({crop_x},{crop_y})")

    filters.append(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=bicubic")
    return filters


def _generate_subtitle_file(segment: ClipSegment, output_path: Path, log_prefix: str) -> Path | None:
    """Generate the Hormozi-style ASS subtitle file for a segment, if any words/hook."""
    hook_text = (segment.hook_reason or "").strip()
    is_real_hook = hook_text and len(hook_text) > 5 and not hook_text.startswith("audio=")
    if not is_real_hook:
        hook_text = ""

    logger.info(
        f"{log_prefix}Subtitle words: {len(segment.words)}, hook: \"{hook_text[:50]}\""
    )
    if not (segment.words or hook_text):
        return None

    sub_path = output_path.parent / f"{output_path.stem}.ass"
    words_to_hormozi_ass(
        segment.words, sub_path,
        words_per_chunk=SUBTITLE_WORDS_PER_CHUNK,
        uppercase=SUBTITLE_UPPERCASE,
        hook_text=hook_text,
    )
    logger.info(
        f"{log_prefix}ASS written: {sub_path} "
        f"(exists={sub_path.exists()}, size={sub_path.stat().st_size}B)"
    )
    return sub_path


def _format_subtitles_filter(sub_path: Path) -> str:
    """Build the FFmpeg subtitles=... filter argument with proper escaping."""
    try:
        sub_rel = sub_path.relative_to(Path.cwd())
    except ValueError:
        sub_rel = sub_path
    sub_escaped = str(sub_rel).replace("\\", "/")
    return f"subtitles={sub_escaped}"


# ── Public entry point ──────────────────────────────────────────────


def render_clip(
    source_path: Path,
    segment: ClipSegment,
    output_path: Path,
    src_width: int = 1920,
    src_height: int = 1080,
    job_id: str = "",
    dubbed_audio_path: Path | None = None,
    render_preset: RenderPreset = RenderPreset.DEFAULT,
    music_path: Path | None = None,
) -> Path:
    """Render a single 9:16 clip from a landscape source video.

    Dispatches to the appropriate preset implementation.
    """
    if render_preset == RenderPreset.PODCAST_BW:
        return _render_clip_podcast_bw(
            source_path=source_path,
            segment=segment,
            output_path=output_path,
            src_width=src_width,
            src_height=src_height,
            job_id=job_id,
            music_path=music_path,
        )

    return _render_clip_default(
        source_path=source_path,
        segment=segment,
        output_path=output_path,
        src_width=src_width,
        src_height=src_height,
        job_id=job_id,
        dubbed_audio_path=dubbed_audio_path,
    )


# ── Default preset (existing behavior) ──────────────────────────────


def _render_clip_default(
    source_path: Path,
    segment: ClipSegment,
    output_path: Path,
    src_width: int,
    src_height: int,
    job_id: str,
    dubbed_audio_path: Path | None,
) -> Path:
    """Default render: dynamic crop 9:16 + Hormozi subtitles + loudnorm.

    Pipeline:
    1. Trim to segment time range
    2. Crop a vertical window centered on average face position
    3. Scale to 1080x1920
    4. Normalize audio (loudnorm)
    5. Burn in subtitles
    6. Output H.264 + AAC mp4
    """
    duration = segment.end - segment.start
    prefix = f"[{job_id}] " if job_id else ""
    logger.info(
        f"{prefix}Rendering clip (default): "
        f"{segment.start:.1f}s → {segment.end:.1f}s ({duration:.1f}s)"
    )

    real_w = src_width
    real_h = src_height
    logger.info(f"{prefix}Source video: {real_w}x{real_h}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_path = _generate_subtitle_file(segment, output_path, prefix)

    filters = _build_crop_scale_filters(segment, real_w, real_h, duration, prefix)
    filters.append("format=yuv420p")

    if sub_path and sub_path.exists():
        sub_filter = _format_subtitles_filter(sub_path)
        logger.info(f"{prefix}Subtitle filter: {sub_filter}")
        filters.append(sub_filter)
    else:
        logger.warning(
            f"{prefix}NO subtitle filter added! sub_path={sub_path}, "
            f"exists={sub_path.exists() if sub_path else 'N/A'}"
        )

    vf = ",".join(filters)

    use_dubbed = dubbed_audio_path and Path(str(dubbed_audio_path)).exists()

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-threads", "2",
        "-ss", f"{segment.start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_path),
    ]

    if use_dubbed:
        cmd.extend(["-i", str(dubbed_audio_path)])
        cmd.extend(["-map", "0:v", "-map", "1:a"])
        logger.info(f"{prefix}Using dubbed audio: {dubbed_audio_path}")

    cmd.extend([
        "-vf", vf,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path).replace("\\", "/"),
    ])

    logger.info(f"{prefix}FFmpeg command:\n  {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"{prefix}FFmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"FFmpeg failed (exit code {result.returncode}): {result.stderr[-500:]}"
        )

    if sub_path and sub_path.exists():
        sub_path.unlink()

    if use_dubbed:
        try:
            Path(str(dubbed_audio_path)).unlink(missing_ok=True)
        except Exception:
            pass

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"{prefix}Clip rendered: {output_path.name} ({file_size_mb:.1f} MB)")
    return output_path


# ── Podcast B&W preset ──────────────────────────────────────────────


def _extract_cropped_clip(
    source_path: Path,
    segment: ClipSegment,
    output_path: Path,
    src_width: int,
    src_height: int,
    log_prefix: str,
) -> Path:
    """Extract a 9:16 cropped + scaled clip (no audio, no subtitles).

    This is the input to the matting stage of the podcast preset.
    """
    duration = segment.end - segment.start
    filters = _build_crop_scale_filters(segment, src_width, src_height, duration, log_prefix)
    filters.append("format=yuv420p")
    vf = ",".join(filters)

    cmd = [
        FFMPEG_BIN, "-y",
        "-loglevel", "error",
        "-threads", "2",
        "-ss", f"{segment.start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_path),
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output_path).replace("\\", "/"),
    ]
    logger.info(f"{log_prefix}Extracting cropped clip → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"{log_prefix}FFmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"Cropped extract failed (rc={result.returncode}): {result.stderr[-300:]}"
        )
    return output_path


def _render_clip_podcast_bw(
    source_path: Path,
    segment: ClipSegment,
    output_path: Path,
    src_width: int,
    src_height: int,
    job_id: str,
    music_path: Path | None,
) -> Path:
    """Podcast B&W render: matting on black + B&W + (optional) music + subtitles.

    Pipeline:
        A. Extract cropped 9:16 clip (no audio).
        B. Run RVM matting: composite foreground on black, output as grayscale video.
        C. Build mixed audio (voice + ducked music) or voice-only loudnorm.
        D. Combine matted video + audio + burn-in subtitles → final mp4.
    """
    from video_maker.matting import matte_clip_to_bw_on_black
    from video_maker.audio_mix import mix_voice_with_music, extract_voice_only

    duration = segment.end - segment.start
    prefix = f"[{job_id}] " if job_id else ""
    logger.info(
        f"{prefix}Rendering clip (podcast_bw): "
        f"{segment.start:.1f}s → {segment.end:.1f}s ({duration:.1f}s)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f".tmp_{output_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    cropped = work_dir / "cropped.mp4"
    matted = work_dir / "matted_bw.mp4"
    audio = work_dir / "audio.m4a"

    sub_path: Path | None = None

    try:
        # ── Step A: extract 9:16 cropped clip ────────────────────────
        _extract_cropped_clip(source_path, segment, cropped, src_width, src_height, prefix)

        # ── Step B: matting (RVM) → fg on black, grayscale ───────────
        logger.info(f"{prefix}Running RVM matting...")
        matte_clip_to_bw_on_black(cropped, matted)

        # ── Step C: audio mixing ─────────────────────────────────────
        if music_path and Path(music_path).exists():
            logger.info(f"{prefix}Mixing audio with music: {music_path.name}")
            mix_voice_with_music(
                source_video=source_path,
                music_path=music_path,
                output_audio=audio,
                clip_start=segment.start,
                clip_duration=duration,
            )
        else:
            if music_path:
                logger.warning(f"{prefix}music_path provided but missing: {music_path}")
            else:
                logger.info(f"{prefix}No music provided — voice only")
            extract_voice_only(source_path, audio, segment.start, duration)

        # ── Step D: build final video (matted + audio + subs) ────────
        sub_path = _generate_subtitle_file(segment, output_path, prefix)

        filters = ["format=yuv420p"]
        if sub_path and sub_path.exists():
            filters.append(_format_subtitles_filter(sub_path))
            logger.info(f"{prefix}Adding subtitles to final composition")
        # Slight contrast/brightness boost — B&W often looks washed out otherwise
        filters.append("eq=contrast=1.12:brightness=0.02")

        vf = ",".join(filters)

        cmd = [
            FFMPEG_BIN, "-y",
            "-threads", "2",
            "-i", str(matted),
            "-i", str(audio),
            "-map", "0:v",
            "-map", "1:a",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-movflags", "+faststart",
            str(output_path).replace("\\", "/"),
        ]
        logger.info(f"{prefix}Final FFmpeg command:\n  {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"{prefix}FFmpeg stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(
                f"Final composition failed (rc={result.returncode}): {result.stderr[-300:]}"
            )

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"{prefix}Clip rendered (podcast_bw): {output_path.name} ({file_size_mb:.1f} MB)")
        return output_path

    finally:
        # Cleanup temp work dir + ASS file
        if sub_path and sub_path.exists():
            try:
                sub_path.unlink()
            except Exception:
                pass
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
