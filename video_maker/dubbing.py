"""Voice dubbing: translate non-French audio → French TTS → mix with original."""

import asyncio
import subprocess
from pathlib import Path
from typing import List

from video_maker.config import FFMPEG_DIR
from video_maker.models import SubtitleWord
from video_maker.utils import logger

import sys as _sys

_FFMPEG = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))
_FFPROBE = str(FFMPEG_DIR / ("ffprobe.exe" if _sys.platform == "win32" else "ffprobe"))

# TTS config
_FR_VOICE = "fr-FR-HenriNeural"
_ORIGINAL_VOLUME = 0.15  # duck original to 15% behind the French voice


# ── Word grouping ─────────────────────────────────────────────────────


def _group_words_into_segments(
    words: List[SubtitleWord],
    max_gap: float = 0.8,
    max_words: int = 15,
    min_words: int = 3,
) -> list[dict]:
    """Group subtitle words into translator-friendly sentence chunks."""
    if not words:
        return []

    segments: list[dict] = []
    current: list[SubtitleWord] = [words[0]]

    for w in words[1:]:
        prev = current[-1]
        gap = w.start - prev.end
        is_sentence_end = prev.word.rstrip().endswith((".", "!", "?"))

        if gap > max_gap or len(current) >= max_words or (is_sentence_end and len(current) >= min_words):
            segments.append({
                "start": current[0].start,
                "end": current[-1].end,
                "text": " ".join(sw.word for sw in current),
            })
            current = [w]
        else:
            current.append(w)

    if current:
        segments.append({
            "start": current[0].start,
            "end": current[-1].end,
            "text": " ".join(sw.word for sw in current),
        })

    return segments


# ── Translation ───────────────────────────────────────────────────────


def _translate_segments(segments: list[dict], source_lang: str) -> list[dict]:
    """Translate segment texts to French via Google Translate."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=source_lang, target="fr")

    for seg in segments:
        try:
            translated = translator.translate(seg["text"])
            seg["text_fr"] = translated if translated else seg["text"]
        except Exception as e:
            logger.warning(f"Translation failed for '{seg['text'][:50]}…': {e}")
            seg["text_fr"] = seg["text"]

    return segments


def translate_words(words: List[SubtitleWord], source_lang: str) -> List[SubtitleWord]:
    """Translate subtitle words to French, preserving approximate timestamps.

    Groups words into sentence chunks, translates each chunk, then
    redistributes timestamps proportionally among the translated words.
    """
    if not words or source_lang == "fr":
        return words

    segments = _group_words_into_segments(words)
    segments = _translate_segments(segments, source_lang)

    translated: list[SubtitleWord] = []
    for seg in segments:
        fr_words = seg["text_fr"].split()
        if not fr_words:
            continue

        seg_duration = max(seg["end"] - seg["start"], 0.1)
        word_dur = seg_duration / len(fr_words)

        for i, w in enumerate(fr_words):
            translated.append(SubtitleWord(
                start=round(seg["start"] + i * word_dur, 3),
                end=round(seg["start"] + (i + 1) * word_dur, 3),
                word=w,
            ))

    logger.info(f"Translated {len(words)} words → {len(translated)} French words")
    return translated


# ── TTS generation ────────────────────────────────────────────────────


async def _generate_tts(text: str, output_path: Path, voice: str = _FR_VOICE):
    """Generate a single TTS audio file via edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def _get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    cmd = [
        _FFPROBE, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _build_atempo_chain(speed: float) -> str:
    """Build atempo filter for arbitrary speed (FFmpeg atempo range: 0.5–2.0)."""
    if speed <= 0:
        return "atempo=1.0"
    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.01:
        parts.append(f"atempo={remaining:.3f}")
    return ",".join(parts) if parts else "atempo=1.0"


# ── Main dubbing pipeline ────────────────────────────────────────────


def create_dubbed_audio(
    words: List[SubtitleWord],
    source_lang: str,
    clip_start: float,
    clip_duration: float,
    source_path: Path,
    work_dir: Path,
) -> Path | None:
    """Full dubbing pipeline: translate → TTS → speed-adjust → mix with ducked original.

    Returns path to the mixed dubbed WAV, or None on failure.
    """
    if not words:
        logger.warning("Dubbing: no words to dub")
        return None

    try:
        # ── 1. Group + translate ──────────────────────────────────────
        segments = _group_words_into_segments(words)
        segments = _translate_segments(segments, source_lang)
        logger.info(f"Dubbing: {len(segments)} segments translated ({source_lang}→fr)")

        # ── 2. Generate TTS for each segment ──────────────────────────
        tts_info: list[dict] = []

        async def _gen_all():
            for i, seg in enumerate(segments):
                text_fr = seg.get("text_fr", "").strip()
                if not text_fr:
                    continue
                tts_path = work_dir / f"_tts_{i:03d}.mp3"
                await _generate_tts(text_fr, tts_path)
                if tts_path.exists() and tts_path.stat().st_size > 0:
                    tts_info.append({
                        "path": tts_path,
                        "start": seg["start"],
                        "end": seg["end"],
                    })

        asyncio.run(_gen_all())
        logger.info(f"Dubbing: {len(tts_info)} TTS segments generated")

        if not tts_info:
            return None

        # ── 3. Build dubbed track (TTS segments placed on timeline) ───
        # Input 0 = silence base; inputs 1..N = TTS files
        inputs = [
            "-f", "lavfi", "-t", f"{clip_duration:.3f}",
            "-i", "anullsrc=r=44100:cl=mono",
        ]
        filter_parts: list[str] = []
        valid_count = 0

        for i, tts in enumerate(tts_info):
            tts_dur = _get_audio_duration(tts["path"])
            if tts_dur <= 0:
                continue

            target_dur = tts["end"] - tts["start"]
            if target_dur < 0.1:
                target_dur = tts_dur  # no speed change needed

            speed = tts_dur / target_dur
            speed = max(0.7, min(1.8, speed))  # keep natural-sounding range

            delay_ms = max(0, int(tts["start"] * 1000))
            input_idx = valid_count + 1
            inputs.extend(["-i", str(tts["path"])])

            atempo = _build_atempo_chain(speed)
            if abs(speed - 1.0) > 0.05:
                filter_parts.append(
                    f"[{input_idx}]{atempo},adelay={delay_ms}|{delay_ms}[s{valid_count}]"
                )
            else:
                filter_parts.append(
                    f"[{input_idx}]adelay={delay_ms}|{delay_ms}[s{valid_count}]"
                )
            valid_count += 1

        if valid_count == 0:
            return None

        # Mix silence base + all delayed TTS segments
        mix_labels = "[0]" + "".join(f"[s{i}]" for i in range(valid_count))
        filter_parts.append(
            f"{mix_labels}amix=inputs={valid_count + 1}:duration=first:normalize=0[dub]"
        )

        dubbed_tts_path = work_dir / "_dubbed_tts.wav"
        cmd_tts = [_FFMPEG, "-y"] + inputs + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[dub]",
            "-c:a", "pcm_s16le", "-ar", "44100",
            str(dubbed_tts_path),
        ]

        r = subprocess.run(cmd_tts, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            logger.error(f"Dubbing TTS mix failed:\n{r.stderr[-500:]}")
            return None
        logger.info(f"Dubbing: TTS track created ({dubbed_tts_path.stat().st_size // 1024} KB)")

        # ── 4. Extract original audio for this clip ───────────────────
        orig_audio = work_dir / "_orig_clip_audio.wav"
        cmd_orig = [
            _FFMPEG, "-y",
            "-ss", f"{clip_start:.3f}", "-t", f"{clip_duration:.3f}",
            "-i", str(source_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            str(orig_audio),
        ]
        subprocess.run(cmd_orig, capture_output=True, text=True, timeout=120, check=True)

        # ── 5. Mix: original (ducked) + French TTS ───────────────────
        mixed_path = work_dir / "_dubbed_mixed.wav"
        cmd_mix = [
            _FFMPEG, "-y",
            "-i", str(orig_audio),
            "-i", str(dubbed_tts_path),
            "-filter_complex",
            f"[0:a]volume={_ORIGINAL_VOLUME}[orig];"
            f"[1:a]volume=1.0[dub];"
            f"[orig][dub]amix=inputs=2:duration=first:normalize=0[out]",
            "-map", "[out]",
            "-c:a", "pcm_s16le", "-ar", "44100",
            str(mixed_path),
        ]
        r = subprocess.run(cmd_mix, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            logger.error(f"Dubbing audio mix failed:\n{r.stderr[-500:]}")
            return None

        # Cleanup intermediate files
        for tts in tts_info:
            tts["path"].unlink(missing_ok=True)
        orig_audio.unlink(missing_ok=True)
        dubbed_tts_path.unlink(missing_ok=True)

        size_kb = mixed_path.stat().st_size // 1024
        logger.info(f"Dubbing: mixed audio ready ({size_kb} KB)")
        return mixed_path

    except Exception as e:
        logger.error(f"Dubbing failed: {e}")
        return None
