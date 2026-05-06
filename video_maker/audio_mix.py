"""Audio mixing helpers for the podcast preset.

Pipeline:
    1. Extract the voice track from the source clip (already trimmed by upstream FFmpeg).
    2. Loop the music file to the clip duration.
    3. Sidechain-compress the music using the voice track (ducking).
    4. Mix voice + ducked music + loudnorm.
    5. Output a single AAC m4a file.
"""

import subprocess
import sys as _sys
from pathlib import Path

from video_maker.config import FFMPEG_DIR
from video_maker.utils import logger

FFMPEG_BIN = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))


def mix_voice_with_music(
    source_video: Path,
    music_path: Path,
    output_audio: Path,
    clip_start: float,
    clip_duration: float,
    music_volume: float = 0.25,
) -> Path:
    """Build a mixed audio track (voice + ducked background music).

    Args:
        source_video: Source video file (we'll pull its audio track).
        music_path: Background music file (any common format).
        output_audio: Destination .m4a file.
        clip_start: Trim point in seconds within the source video.
        clip_duration: Length of the resulting audio in seconds.
        music_volume: Base music gain (0.0–1.0) before sidechain compression.

    Returns:
        Path to output_audio.
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    # filter_complex:
    #   [0:a] = voice extracted from the source video at clip_start for clip_duration
    #   [1:a] = music looped to clip_duration, attenuated to music_volume
    #   [music][voice] sidechaincompress → [ducked]
    #   [voice][ducked] amix → [mixed]
    #   [mixed] loudnorm → [out]
    filter_complex = (
        f"[0:a]aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[voice_raw];"
        f"[1:a]aformat=channel_layouts=stereo,"
        f"aloop=loop=-1:size=2147483647,"
        f"atrim=duration={clip_duration:.3f},"
        f"asetpts=PTS-STARTPTS,"
        f"volume={music_volume}[music];"
        f"[music][voice_raw]sidechaincompress="
        f"threshold=0.03:ratio=8:attack=5:release=300:makeup=1[ducked];"
        f"[voice_raw][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )

    cmd = [
        FFMPEG_BIN, "-y",
        "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-t", f"{clip_duration:.3f}",
        "-i", str(source_video),
        "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        str(output_audio).replace("\\", "/"),
    ]

    logger.info(f"[audio_mix] Mixing voice + music ({clip_duration:.1f}s, music_vol={music_volume})")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        logger.error(f"[audio_mix] FFmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Audio mix failed (rc={result.returncode}): {result.stderr[-300:]}")

    logger.info(f"[audio_mix] Done: {output_audio.name} ({output_audio.stat().st_size / 1024:.0f} KB)")
    return output_audio


def extract_voice_only(
    source_video: Path,
    output_audio: Path,
    clip_start: float,
    clip_duration: float,
) -> Path:
    """Fallback when no music is provided: just extract + loudnorm the voice."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BIN, "-y",
        "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-t", f"{clip_duration:.3f}",
        "-i", str(source_video),
        "-vn",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        str(output_audio).replace("\\", "/"),
    ]
    logger.info(f"[audio_mix] Voice-only extract ({clip_duration:.1f}s)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.error(f"[audio_mix] FFmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Voice extract failed (rc={result.returncode}): {result.stderr[-300:]}")
    return output_audio
