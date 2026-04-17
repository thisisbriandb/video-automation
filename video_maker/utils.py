"""Utility functions: temp file management, SRT generation, helpers"""

import logging
import shutil
from pathlib import Path
from video_maker.models import SubtitleLine, SubtitleWord

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


def words_to_hormozi_srt(
    words: list[SubtitleWord],
    output_path: Path,
    words_per_chunk: int = 3,
    uppercase: bool = True,
) -> Path:
    """Generate Hormozi-style SRT: 2-3 words per subtitle, big and punchy.

    Groups words into chunks of `words_per_chunk` with tight timing.
    """
    if not words:
        output_path.write_text("", encoding="utf-8")
        return output_path

    chunks: list[SubtitleLine] = []
    i = 0
    while i < len(words):
        group = words[i : i + words_per_chunk]
        text = " ".join(w.word for w in group)
        if uppercase:
            text = text.upper()
        chunks.append(SubtitleLine(
            start=group[0].start,
            end=group[-1].end,
            text=text,
        ))
        i += words_per_chunk

    # Ensure no gap between chunks (avoids flicker)
    for j in range(len(chunks) - 1):
        if chunks[j].end < chunks[j + 1].start:
            chunks[j].end = chunks[j + 1].start

    generate_srt(chunks, output_path)
    logger.info(f"Generated Hormozi SRT: {output_path.name} ({len(chunks)} chunks from {len(words)} words)")
    return output_path


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.CC (centiseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds % 1) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def words_to_hormozi_ass(
    words: list[SubtitleWord],
    output_path: Path,
    words_per_chunk: int = 3,
    uppercase: bool = True,
    hook_text: str = "",
) -> Path:
    """Generate Hormozi-style ASS subtitle file with embedded styles.

    Uses ASS format so FFmpeg subtitles filter needs NO force_style parameter,
    avoiding all filter-chain escaping issues on Windows.
    The hook text (if provided) is rendered as a separate style at the top of
    the screen with a fade-in/fade-out animation — no drawtext filter needed.
    """
    # ASS header with embedded styles
    # - Default: bottom-center subtitles (Alignment=2)
    # - Hook: top-center hook phrase (Alignment=8), bigger font, fade effect
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Impact,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,2,10,10,180,1\n"
        "Style: Hook,Impact,100,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,3,0,0,5,40,40,0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]

    # Hook line (center screen, 0s → 4s, no fade-in, fade-out 800ms)
    if hook_text:
        lines.append(
            f"Dialogue: 1,0:00:00.00,0:00:04.00,Hook,,0,0,0,,"
            f"{{\\fad(0,800)}}{hook_text}\n"
        )

    if not words:
        output_path.write_text("".join(lines), encoding="utf-8-sig")
        return output_path

    # Group words into chunks
    chunks: list[SubtitleLine] = []
    i = 0
    while i < len(words):
        group = words[i : i + words_per_chunk]
        text = " ".join(w.word for w in group)
        if uppercase:
            text = text.upper()
        chunks.append(SubtitleLine(
            start=group[0].start,
            end=group[-1].end,
            text=text,
        ))
        i += words_per_chunk

    # Ensure no gap between chunks (avoids flicker)
    for j in range(len(chunks) - 1):
        if chunks[j].end < chunks[j + 1].start:
            chunks[j].end = chunks[j + 1].start

    # Build dialogue lines
    for chunk in chunks:
        start_ts = _format_ass_time(chunk.start)
        end_ts = _format_ass_time(chunk.end)
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{chunk.text}\n")

    output_path.write_text("".join(lines), encoding="utf-8-sig")
    logger.info(f"Generated Hormozi ASS: {output_path.name} ({len(chunks)} chunks from {len(words)} words, hook={'yes' if hook_text else 'no'})")
    return output_path
