"""FFmpeg renderer: crop, scale, subtitle burn-in, audio normalization."""

import json
import subprocess
from pathlib import Path
from video_maker.config import FFMPEG_DIR, SUBTITLE_WORDS_PER_CHUNK, SUBTITLE_UPPERCASE
from video_maker.models import ClipSegment, FaceKeyframe
from video_maker.utils import logger, words_to_hormozi_srt, clamp

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


def _build_subtitle_style() -> str:
    """Return ASS-style override for Hormozi-style subtitles.
    
    Note: ASS default PlayResY=288. All MarginV/FontSize values are in that
    coordinate space, NOT in actual video pixels.
    """
    return (
        "FontName=Impact,"
        "FontSize=16,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=35"
    )


def render_clip(
    source_path: Path,
    segment: ClipSegment,
    output_path: Path,
    src_width: int = 1920,
    src_height: int = 1080,
    job_id: str = "",
) -> Path:
    """Render a single 9:16 clip from a landscape source video.

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
    logger.info(f"{prefix}Rendering clip: {segment.start:.1f}s → {segment.end:.1f}s ({duration:.1f}s)")

    # ── Use provided source dimensions ──────────────────────────────
    real_w = src_width
    real_h = src_height
    logger.info(f"{prefix}Source video: {real_w}x{real_h}")

    # ── Compute crop window (9:16 from source) ─────────────────────
    crop_h = real_h
    crop_w = int(real_h * 9 / 16)

    # If source is narrower than 9:16 crop, use full width
    if crop_w > real_w:
        crop_w = real_w
        crop_h = int(real_w * 16 / 9)

    # Make even (FFmpeg requires even dimensions for libx264)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    crop_y = int(clamp((real_h - crop_h) // 2, 0, real_h - crop_h))

    # ── Generate Hormozi-style SRT file ─────────────────────────────
    srt_path = None
    logger.info(f"{prefix}Subtitle words: {len(segment.words)}")
    if segment.words:
        srt_path = output_path.parent / f"{output_path.stem}.srt"
        words_to_hormozi_srt(segment.words, srt_path, words_per_chunk=SUBTITLE_WORDS_PER_CHUNK, uppercase=SUBTITLE_UPPERCASE)
        logger.info(f"{prefix}SRT written: {srt_path} (exists={srt_path.exists()}, size={srt_path.stat().st_size}B)")

    # ── Build filter chain ──────────────────────────────────────────
    filters = []

    # Dynamic crop: interpolate face X between keyframes for smooth tracking
    crop_x_expr = _build_dynamic_crop_x(
        segment.face_keyframes, crop_w, real_w, duration
    )
    if crop_x_expr:
        filters.append(f"crop={crop_w}:{crop_h}:{crop_x_expr}:{crop_y}")
        logger.info(f"{prefix}Dynamic crop: {crop_w}x{crop_h}, {len(segment.face_keyframes)} keyframes, y={crop_y}")
    else:
        # Fallback: static crop at average face position
        face_x = _average_face_x(segment.face_keyframes, real_w // 2)
        if not segment.face_keyframes:
            logger.warning(f"{prefix}No face detected — centering crop")
        crop_x = int(clamp(face_x - crop_w // 2, 0, real_w - crop_w))
        filters.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
        logger.info(f"{prefix}Static crop: {crop_w}x{crop_h} at ({crop_x},{crop_y})")

    # Scale to 1080x1920 (use bicubic for better speed/quality balance than lanczos on CPU)
    filters.append(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=bicubic")

    # Pixel format (must come before subtitles)
    filters.append("format=yuv420p")

    # Burn subtitles (after scale so coords match output resolution)
    # Use relative path to avoid Windows drive-letter colon breaking FFmpeg filter parser
    if srt_path and srt_path.exists():
        try:
            srt_rel = srt_path.relative_to(Path.cwd())
        except ValueError:
            srt_rel = srt_path
        srt_escaped = str(srt_rel).replace("\\", "/")
        style = _build_subtitle_style()
        sub_filter = f"subtitles={srt_escaped}:force_style='{style}'"
        logger.info(f"{prefix}Subtitle filter: {sub_filter[:150]}")
        filters.append(sub_filter)
    else:
        logger.warning(f"{prefix}NO subtitle filter added! srt_path={srt_path}, exists={srt_path.exists() if srt_path else 'N/A'}")

    vf = ",".join(filters)

    # ── Build FFmpeg command ────────────────────────────────────────
    # Note: -threads 2 is a safe limit to avoid oversubscription when running parallel workers
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-threads", "2",
        "-ss", f"{segment.start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_path),
        "-vf", vf,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"{prefix}FFmpeg command:\n  {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error(f"{prefix}FFmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"FFmpeg failed (exit code {result.returncode}): {result.stderr[-500:]}")

    # Clean up SRT
    if srt_path and srt_path.exists():
        srt_path.unlink()

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"{prefix}Clip rendered: {output_path.name} ({file_size_mb:.1f} MB)")
    return output_path
