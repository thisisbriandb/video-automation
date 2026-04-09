"""Local-first video analyzer: score → rank → expand → transcribe final clips only."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from video_maker.config import MAX_CLIPS, CLIP_MIN_DURATION, NUM_WORKERS, WHISPER_MODEL
from video_maker.models import (
    AnalysisResult,
    ClipSegment,
    ScoredSegment,
    SubtitleWord,
)
from video_maker.scorer import (
    score_video,
    _expand_to_min_duration,
    _merge_overlapping,
)
from video_maker.transcriber import (
    transcribe_segment_from_file,
    parse_youtube_json3,
    extract_words_for_clip,
)
from video_maker.utils import logger


def analyze_video(
    video_path: Path,
    work_dir: Path,
    subs_path: Path | None = None,
) -> AnalysisResult:
    """Full local analysis pipeline: score → expand → batch-transcribe final clips.

    Skips the first Whisper pass (text_score re-ranking) — audio+visual scores
    are sufficient for segment selection. Whisper runs only once on final clips.
    Audio segments are pre-extracted in parallel for maximum throughput.

    Args:
        video_path: Path to the high-res video file.
        work_dir: Temporary working directory for this job.

    Returns:
        AnalysisResult with up to MAX_CLIPS ClipSegments, each ≥ CLIP_MIN_DURATION.
    """

    # ── Step 1: Local scoring (audio + visual) ────────────────────────
    top_segments, audio_path, duration = score_video(video_path, work_dir)
    logger.info(f"Pre-scoring complete: {len(top_segments)} candidate segments")

    if not top_segments:
        logger.warning("No scorable segments found.")
        return AnalysisResult(clips=[])

    # ── Step 2: Merge overlapping, expand to ≥60s (no Whisper re-ranking)
    merged = _merge_overlapping(top_segments)
    expanded = _expand_to_min_duration(merged, duration)
    logger.info(f"After merge+expand: {len(expanded)} segments ≥ {CLIP_MIN_DURATION}s")

    # ── Step 3: Take top MAX_CLIPS ────────────────────────────────────
    expanded.sort(key=lambda s: s.total_score, reverse=True)
    final_segments = expanded[:MAX_CLIPS]
    n_clips = len(final_segments)

    # ── Step 4+5: Get subtitles (YouTube subs → instant, Whisper → slow)
    t_subs = time.time()

    if subs_path and subs_path.exists():
        # ── Fast path: YouTube JSON3 word-level subtitles ──────────
        logger.info(f"Using YouTube subtitles from {subs_path.name} (skipping Whisper)")
        all_words = parse_youtube_json3(subs_path)

        clips = []
        for rank, seg in enumerate(final_segments):
            words = extract_words_for_clip(all_words, seg.start, seg.end)
            virality = max(1, min(10, int(seg.total_score * 10)))
            clip_dur = seg.end - seg.start

            clips.append(ClipSegment(
                start=seg.start,
                end=seg.end,
                virality_score=virality,
                title=f"Clip #{rank + 1} ({clip_dur:.0f}s)",
                hook_reason=f"audio={seg.audio_score:.2f} visual={seg.visual_score:.2f}",
                words=words,
            ))
            logger.info(f"  Clip {rank + 1}: {len(words)} words from YouTube subs")

        logger.info(f"YouTube subtitle extraction done in {time.time() - t_subs:.1f}s")
    else:
        # ── Slow path: Whisper transcription ───────────────────────
        logger.info(f"No YouTube subs available — falling back to Whisper (model: {WHISPER_MODEL})")

        logger.info(f"Pre-extracting {n_clips} audio segments with {NUM_WORKERS} workers...")
        t_extract = time.time()

        from video_maker.transcriber import _extract_segment_audio, FFMPEG_BIN
        audio_files: dict[int, Path] = {}

        def _extract_one(rank_seg):
            rank, seg = rank_seg
            out = work_dir / f"final_audio_{rank:02d}.wav"
            _extract_segment_audio(audio_path, seg.start, seg.end - seg.start, out)
            return rank, out

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            for rank, out_path in pool.map(_extract_one, enumerate(final_segments)):
                audio_files[rank] = out_path

        logger.info(f"Audio extraction done ({time.time() - t_extract:.1f}s)")

        # Total Whisper budget: 10 minutes max for all clips combined
        WHISPER_TOTAL_BUDGET = int(__import__('os').environ.get('WHISPER_TOTAL_BUDGET', '600'))
        logger.info(f"Transcribing {n_clips} final clips with Whisper (budget: {WHISPER_TOTAL_BUDGET}s)...")
        transcriptions: dict[int, list] = {}

        t_whisper = time.time()
        for rank, path in audio_files.items():
            elapsed = time.time() - t_whisper
            remaining = WHISPER_TOTAL_BUDGET - elapsed
            if remaining < 30:
                logger.warning(
                    f"Whisper budget exhausted ({elapsed:.0f}s / {WHISPER_TOTAL_BUDGET}s) "
                    f"— skipping remaining {len(audio_files) - len(transcriptions)} clips"
                )
                break
            transcriptions[rank] = transcribe_segment_from_file(path)
            logger.info(f"  [{rank + 1}/{n_clips}] {len(transcriptions[rank])} words ({time.time() - t_whisper:.0f}s elapsed)")

        clips = []
        for rank, seg in enumerate(final_segments):
            words = transcriptions.get(rank, [])
            virality = max(1, min(10, int(seg.total_score * 10)))
            clip_dur = seg.end - seg.start

            clips.append(ClipSegment(
                start=seg.start,
                end=seg.end,
                virality_score=virality,
                title=f"Clip #{rank + 1} ({clip_dur:.0f}s)",
                hook_reason=f"audio={seg.audio_score:.2f} visual={seg.visual_score:.2f}",
                words=words,
            ))

            # Cleanup temp audio
            audio_file = audio_files.get(rank)
            if audio_file and audio_file.exists():
                audio_file.unlink()

        n_with_subs = sum(1 for c in clips if c.words)
        logger.info(
            f"Whisper done: {n_with_subs}/{n_clips} clips have subtitles "
            f"({time.time() - t_subs:.1f}s total)"
        )

    clips.sort(key=lambda c: c.start)
    logger.info(f"Analysis complete: {len(clips)} clips ready for rendering")
    return AnalysisResult(clips=clips)
