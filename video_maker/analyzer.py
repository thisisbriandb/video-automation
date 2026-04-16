"""Video analyzer: Gemini-first (smart segmentation) with scoring fallback."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from video_maker.config import (
    MAX_CLIPS, CLIP_MIN_DURATION, NUM_WORKERS, WHISPER_MODEL, GEMINI_API_KEY,
)
from video_maker.models import (
    AnalysisResult,
    ClipSegment,
    ScoredSegment,
)
from video_maker.scorer import (
    score_video,
    _expand_to_min_duration,
    _merge_overlapping,
)
from video_maker.transcriber import transcribe_segment_from_file, unload_model
from video_maker.utils import logger


def analyze_video(
    video_path: Path,
    work_dir: Path,
    youtube_url: str = "",
) -> AnalysisResult:
    """Analyze video to find the best clips.

    Two modes:
      • Gemini mode (GEMINI_API_KEY set): send YouTube URL to Gemini for
        intelligent segmentation, then Whisper for subtitles only.
      • Scoring mode (fallback): local audio+visual scoring → expand → Whisper.

    Args:
        video_path: Path to the high-res video file.
        work_dir: Temporary working directory for this job.
        youtube_url: Original YouTube URL (used for Gemini mode).

    Returns:
        AnalysisResult with up to MAX_CLIPS ClipSegments.
    """
    if GEMINI_API_KEY and youtube_url:
        return _analyze_with_gemini(video_path, work_dir, youtube_url)
    return _analyze_with_scoring(video_path, work_dir)


# ── Gemini-first path ─────────────────────────────────────────────────


def _analyze_with_gemini(
    video_path: Path,
    work_dir: Path,
    youtube_url: str,
) -> AnalysisResult:
    """Gemini segments the video, Whisper only transcribes selected clips."""
    from video_maker.segmenter import segment_with_gemini

    t0 = time.time()

    # 1. Gemini: analyze video via YouTube URL
    clips = segment_with_gemini(youtube_url)
    if not clips:
        logger.warning("Gemini returned no segments — falling back to scoring")
        return _analyze_with_scoring(video_path, work_dir)

    logger.info(f"Gemini segmentation done ({time.time() - t0:.1f}s), {len(clips)} clips")

    # 2. Whisper: transcribe each Gemini-selected clip for subtitles
    _whisper_transcribe_clips(clips, video_path, work_dir)

    logger.info(f"Analysis complete (Gemini): {len(clips)} clips ready for rendering")
    return AnalysisResult(clips=clips)


# ── Scoring fallback path ─────────────────────────────────────────────


def _analyze_with_scoring(
    video_path: Path,
    work_dir: Path,
) -> AnalysisResult:
    """Original scoring-based analysis (audio + visual + Whisper)."""

    # ── Step 1: Local scoring (audio + visual) ────────────────────
    top_segments, audio_path, duration = score_video(video_path, work_dir)
    logger.info(f"Pre-scoring complete: {len(top_segments)} candidate segments")

    if not top_segments:
        logger.warning("No scorable segments found.")
        return AnalysisResult(clips=[])

    # ── Step 2: Merge overlapping, expand to ≥60s (no Whisper re-ranking)
    merged = _merge_overlapping(top_segments)
    expanded = _expand_to_min_duration(merged, duration)
    logger.info(f"After merge+expand: {len(expanded)} segments ≥ {CLIP_MIN_DURATION}s")

    # ── Step 3: Take top MAX_CLIPS ──────────────────────────────
    expanded.sort(key=lambda s: s.total_score, reverse=True)
    final_segments = expanded[:MAX_CLIPS]

    # ── Step 4: Build ClipSegments ─────────────────────────────
    clips = []
    for rank, seg in enumerate(final_segments):
        clip_dur = seg.end - seg.start
        clips.append(ClipSegment(
            start=seg.start,
            end=seg.end,
            virality_score=max(1, min(10, int(seg.total_score * 10))),
            title=f"Clip #{rank + 1} ({clip_dur:.0f}s)",
            hook_reason=f"audio={seg.audio_score:.2f} visual={seg.visual_score:.2f}",
            words=[],
            language="",
        ))

    # ── Step 5: Whisper transcribe selected clips ───────────────
    _whisper_transcribe_clips(clips, video_path, work_dir, audio_source=audio_path)

    # Free scoring audio
    if audio_path.exists():
        audio_path.unlink()
        logger.info("Freed scoring audio WAV from disk")

    clips.sort(key=lambda c: c.start)
    logger.info(f"Analysis complete (scoring): {len(clips)} clips ready for rendering")
    return AnalysisResult(clips=clips)


# ── Shared: Whisper transcription for selected clips ──────────────────


def _whisper_transcribe_clips(
    clips: List[ClipSegment],
    video_path: Path,
    work_dir: Path,
    audio_source: Path | None = None,
) -> None:
    """Transcribe each clip with Whisper for subtitle generation.

    Modifies clips in-place: fills .words and .language.
    If audio_source is provided, extracts segments from it (faster).
    Otherwise extracts from the video file.
    """
    from video_maker.transcriber import _extract_segment_audio, FFMPEG_BIN
    from collections import Counter

    n_clips = len(clips)
    source = audio_source if audio_source and audio_source.exists() else video_path

    t0 = time.time()
    logger.info(f"Transcribing {n_clips} clips with Whisper (model: {WHISPER_MODEL})")

    # Extract audio segments in parallel
    audio_files: dict[int, Path] = {}

    def _extract_one(args):
        idx, clip = args
        out = work_dir / f"_clip_audio_{idx:02d}.wav"
        _extract_segment_audio(source, clip.start, clip.end - clip.start, out)
        return idx, out

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for idx, out_path in pool.map(_extract_one, enumerate(clips)):
            audio_files[idx] = out_path

    # Transcribe sequentially (model loaded once)
    WHISPER_TOTAL_BUDGET = int(__import__('os').environ.get('WHISPER_TOTAL_BUDGET', '600'))
    detected_languages: list[str] = []
    t_whisper = time.time()

    for idx in range(n_clips):
        path = audio_files.get(idx)
        if not path:
            continue

        elapsed = time.time() - t_whisper
        if elapsed > WHISPER_TOTAL_BUDGET:
            logger.warning(f"Whisper budget exhausted ({elapsed:.0f}s) — skipping remaining clips")
            break

        result = transcribe_segment_from_file(path)
        clips[idx].words = result.words
        if result.words:
            detected_languages.append(result.language)
        logger.info(
            f"  [{idx + 1}/{n_clips}] {len(result.words)} words, "
            f"lang={result.language} ({elapsed:.0f}s elapsed)"
        )

        # Cleanup
        path.unlink(missing_ok=True)

    # Set language on all clips (majority vote)
    video_language = Counter(detected_languages).most_common(1)[0][0] if detected_languages else "fr"
    for clip in clips:
        if not clip.language:
            clip.language = video_language
    logger.info(f"Detected language: {video_language}")

    # Free Whisper
    unload_model()

    n_with_subs = sum(1 for c in clips if c.words)
    logger.info(f"Whisper done: {n_with_subs}/{n_clips} clips have subtitles ({time.time() - t0:.1f}s)")
