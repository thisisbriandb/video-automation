"""Local-first video analyzer: score → rank → expand → transcribe final clips only."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from video_maker.config import MAX_CLIPS, CLIP_MIN_DURATION, NUM_WORKERS
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
from video_maker.transcriber import transcribe_files_parallel
from video_maker.utils import logger


def analyze_video(
    video_path: Path,
    work_dir: Path,
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

    # ── Step 4: Pre-extract all audio segments in parallel ────────────
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

    # ── Step 5: Transcribe all segments (parallel on CPU / sequential on GPU)
    logger.info(f"Transcribing {n_clips} final clips with Whisper...")
    t_whisper = time.time()

    transcriptions = transcribe_files_parallel(audio_files)

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

    logger.info(f"Transcription done: {n_clips} clips in {time.time() - t_whisper:.1f}s")

    clips.sort(key=lambda c: c.start)
    logger.info(f"Analysis complete: {len(clips)} clips ready for rendering")
    return AnalysisResult(clips=clips)
