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


def _build_subtitle_style() -> str:
    """Return ASS-style override for Hormozi-style subtitles: lower third, just below face."""
    return (
        "FontName=Impact,"
        "FontSize=18,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=3,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=550"
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

    # Center crop on average face X position
    face_x = _average_face_x(segment.face_keyframes, real_w // 2)
    # Scale face_x from Gemini's assumed 1920-wide to actual width
    face_x = int(face_x * real_w / 1920)

    half_crop = crop_w // 2
    crop_x = int(clamp(face_x - half_crop, 0, real_w - crop_w))
    crop_y = int(clamp((real_h - crop_h) // 2, 0, real_h - crop_h))

    logger.info(f"{prefix}Crop: {crop_w}x{crop_h} at ({crop_x},{crop_y}), face_x={face_x}")

    # ── Generate Hormozi-style SRT file ─────────────────────────────
    srt_path = None
    if segment.words:
        srt_path = output_path.parent / f"{output_path.stem}.srt"
        words_to_hormozi_srt(segment.words, srt_path, words_per_chunk=SUBTITLE_WORDS_PER_CHUNK, uppercase=SUBTITLE_UPPERCASE)

    # ── Build filter chain ──────────────────────────────────────────
    filters = []

    # Static crop
    filters.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")

    # Scale to 1080x1920 (use bicubic for better speed/quality balance than lanczos on CPU)
    filters.append(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=bicubic")

    # Pixel format (must come before subtitles)
    filters.append("format=yuv420p")

    # Burn subtitles (after scale so coords match output resolution)
    if srt_path and srt_path.exists():
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\\\:")
        style = _build_subtitle_style()
        filters.append(f"subtitles='{srt_escaped}':force_style='{style}'")

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
