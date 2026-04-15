import subprocess
import threading
import torch
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, NamedTuple

from video_maker.config import FFMPEG_DIR, WHISPER_MODEL, NUM_WORKERS
from video_maker.models import SubtitleWord
from video_maker.utils import logger


class TranscriptionResult(NamedTuple):
    """Whisper transcription output: words + auto-detected language."""
    words: List[SubtitleWord]
    language: str  # ISO 639-1 code (e.g. 'fr', 'en', 'es')

# Minimum audio file size (bytes) to attempt transcription — below this, Whisper
# crashes with tensor reshape errors because the mel spectrogram is degenerate.
_MIN_AUDIO_BYTES = 32_000  # ~1 second at 16kHz 16-bit mono

# Max seconds to spend on a single clip transcription before giving up.
# Medium model on CPU needs ~3-5 min per 60s clip; default must be generous.
WHISPER_TIMEOUT = int(__import__('os').environ.get('WHISPER_TIMEOUT', '600'))

import sys as _sys
FFMPEG_BIN = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))

# Load Whisper model (lazy loading — used for sequential / GPU path)
_model = None


def _get_model():
    global _model
    if _model is None:
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model ({WHISPER_MODEL}) on {device}...")
        _model = whisper.load_model(WHISPER_MODEL, device=device)
    return _model


def unload_model():
    """Free the Whisper model from memory (call after transcription is done)."""
    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper model unloaded from memory")


def _extract_segment_audio(
    audio_path: Path, start: float, duration: float, out_path: Path
) -> Path:
    """Extract a segment from the full audio WAV."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(audio_path),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out_path


def transcribe_segment(
    audio_path: Path, start: float, end: float, work_dir: Path
) -> List[SubtitleWord]:
    """Transcribe a single segment of audio using Whisper with word-level timestamps.

    Args:
        audio_path: Path to the full extracted WAV audio.
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        work_dir: Directory for temporary files.

    Returns:
        List of SubtitleWord with word-level timestamps relative to segment start.
    """
    model = _get_model()
    duration = end - start
    logger.info(f"Transcribing segment {start:.1f}s → {end:.1f}s ({duration:.1f}s)")

    temp_audio = work_dir / f"seg_audio_{start:.0f}_{end:.0f}.wav"
    try:
        _extract_segment_audio(audio_path, start, duration, temp_audio)
        result = model.transcribe(
            str(temp_audio),
            task="transcribe",
            language="fr",
            word_timestamps=True,
            verbose=False,
        )

        words: List[SubtitleWord] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                text = w.get("word", "").strip()
                if text:
                    words.append(SubtitleWord(
                        start=round(w["start"], 3),
                        end=round(w["end"], 3),
                        word=text,
                    ))

        logger.info(f"Transcription complete: {len(words)} words")
        return words

    except Exception as e:
        logger.error(f"Transcription failed for {start:.1f}-{end:.1f}s: {e}")
        return []
    finally:
        if temp_audio.exists():
            temp_audio.unlink()


def transcribe_segment_from_file(audio_file: Path) -> TranscriptionResult:
    """Transcribe a pre-extracted audio file directly with Whisper (single model).

    Skips the FFmpeg extraction step — audio_file must already be 16kHz mono WAV.
    Uses a thread-based timeout to prevent infinite hangs on slow CPUs.
    Returns TranscriptionResult(words, language) with auto-detected language.
    """
    # Validate audio file before loading the model
    if not audio_file.exists():
        logger.error(f"Audio file does not exist: {audio_file}")
        return TranscriptionResult(words=[], language="fr")
    fsize = audio_file.stat().st_size
    if fsize < _MIN_AUDIO_BYTES:
        logger.warning(f"Audio file too small ({fsize} bytes), skipping: {audio_file.name}")
        return TranscriptionResult(words=[], language="fr")

    model = _get_model()
    logger.info(f"Transcribing {audio_file.name} ({fsize} bytes, timeout={WHISPER_TIMEOUT}s)...")

    # Run transcription in a thread so we can enforce a timeout
    result_container: list = []
    error_container: list = []

    def _run():
        try:
            r = model.transcribe(
                str(audio_file),
                task="transcribe",
                word_timestamps=True,
                verbose=False,
            )
            result_container.append(r)
        except Exception as e:
            error_container.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=WHISPER_TIMEOUT)

    if t.is_alive():
        logger.warning(f"Whisper TIMEOUT ({WHISPER_TIMEOUT}s) for {audio_file.name} — skipping")
        return TranscriptionResult(words=[], language="fr")

    if error_container:
        logger.error(f"Transcription failed for {audio_file.name}: {error_container[0]}")
        return TranscriptionResult(words=[], language="fr")

    if not result_container:
        logger.error(f"Transcription returned no result for {audio_file.name}")
        return TranscriptionResult(words=[], language="fr")

    result = result_container[0]
    detected_lang = result.get("language", "fr")
    words: List[SubtitleWord] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if text:
                words.append(SubtitleWord(
                    start=round(w["start"], 3),
                    end=round(w["end"], 3),
                    word=text,
                ))

    logger.info(f"Transcription complete: {len(words)} words, language={detected_lang}")
    return TranscriptionResult(words=words, language=detected_lang)


# ── Parallel Whisper (CPU workers) ─────────────────────────────────

_worker_model = None


def _transcribe_one_file_cpu(
    audio_path_str: str, whisper_model_name: str
) -> List[dict]:
    """Transcribe a single audio file in its own process with its own Whisper model.

    Top-level function (pickleable) for ProcessPoolExecutor.
    Returns list of dicts [{start, end, word}, ...] (not Pydantic — must be pickleable).
    """
    global _worker_model
    import whisper as _whisper
    if _worker_model is None:
        _worker_model = _whisper.load_model(whisper_model_name, device="cpu")

    result = _worker_model.transcribe(
        audio_path_str,
        task="transcribe",
        language="fr",
        word_timestamps=True,
        verbose=False,
    )
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if text:
                words.append({"start": round(w["start"], 3), "end": round(w["end"], 3), "word": text})
    return words


def transcribe_files_parallel(
    audio_files: dict[int, Path],
) -> dict[int, List[SubtitleWord]]:
    """Transcribe multiple audio files in parallel.

    - If CUDA is available: sequential on GPU (faster per-file).
    - If CPU only: parallel with ProcessPoolExecutor (each worker loads its own model).

    Args:
        audio_files: dict mapping rank → Path to pre-extracted WAV.

    Returns:
        dict mapping rank → list of SubtitleWord.
    """
    n_files = len(audio_files)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        logger.info(f"Whisper: GPU available — transcribing {n_files} files sequentially on CUDA")
        results: dict[int, List[SubtitleWord]] = {}
        for rank, path in audio_files.items():
            results[rank] = transcribe_segment_from_file(path)
            logger.info(f"  [{rank + 1}/{n_files}] {len(results[rank])} words")
        return results

    # On small containers (Railway etc.), ProcessPoolExecutor spawns N copies of
    # the Whisper model, each ~500MB+.  With ≤2 workers we stay sequential to
    # avoid OOM / silent failures that produce empty word lists (= no subtitles).
    if NUM_WORKERS <= 2:
        logger.info(
            f"Whisper: CPU sequential mode ({n_files} files, model: {WHISPER_MODEL}) "
            f"— NUM_WORKERS={NUM_WORKERS}, skipping ProcessPool to save memory"
        )
        results = {}
        for rank, path in audio_files.items():
            results[rank] = transcribe_segment_from_file(path)
            logger.info(f"  [{rank + 1}/{n_files}] {len(results[rank])} words")
        return results

    # CPU parallel path (only when we have enough workers / RAM)
    n_workers = min(n_files, max(2, NUM_WORKERS // 2))
    logger.info(
        f"Whisper: No GPU — transcribing {n_files} files in parallel "
        f"with {n_workers} CPU workers (model: {WHISPER_MODEL})"
    )

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut_to_rank = {
            pool.submit(
                _transcribe_one_file_cpu, str(audio_files[rank]), WHISPER_MODEL
            ): rank
            for rank in audio_files
        }
        for fut in as_completed(fut_to_rank):
            rank = fut_to_rank[fut]
            try:
                raw_words = fut.result()
                results[rank] = [
                    SubtitleWord(start=w["start"], end=w["end"], word=w["word"])
                    for w in raw_words
                ]
                logger.info(
                    f"  Whisper worker done: file {rank + 1}/{n_files} "
                    f"({len(results[rank])} words)"
                )
            except Exception as e:
                logger.error(f"  Whisper worker failed for rank {rank}: {e}")
                results[rank] = []

    return results


def compute_text_score(words: List[SubtitleWord]) -> float:
    """Compute a text richness score (0-1) from word-level transcription.

    Factors: word density, lexical diversity (unique/total ratio).
    """
    if not words:
        return 0.0

    word_texts = [w.word.lower() for w in words]
    total_words = len(word_texts)
    if total_words == 0:
        return 0.0

    # Lexical diversity: unique words / total words
    unique_ratio = len(set(word_texts)) / total_words

    # Word density: words per second of content
    duration = max(words[-1].end - words[0].start, 1.0)
    density = min(total_words / duration / 3.0, 1.0)  # ~3 words/sec = max density

    return round(0.5 * density + 0.5 * unique_ratio, 4)
