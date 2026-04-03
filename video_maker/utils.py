"""Utility functions: temp file management, SRT generation, helpers"""

import logging
import shutil
from pathlib import Path
from video_maker.models import SubtitleLine

logger = logging.getLogger("video_maker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)


def cleanup_directory(dir_path: Path) -> None:
    """Remove a directory and all its contents."""
    if dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)
        logger.info(f"Cleaned up: {dir_path}")


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(subtitles: list[SubtitleLine], output_path: Path) -> Path:
    """Generate an SRT subtitle file from SubtitleLine list."""
    lines = []
    for i, sub in enumerate(subtitles, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(sub.start)} --> {format_srt_time(sub.end)}")
        lines.append(sub.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated SRT: {output_path} ({len(subtitles)} lines)")
    return output_path


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))
