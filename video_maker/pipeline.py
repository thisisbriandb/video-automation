"""Pipeline orchestrator: download → local scoring → whisper → render (parallel)."""

import asyncio
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from video_maker.config import WORKING_DIR, OUTPUT_DIR, NUM_WORKERS
from video_maker.models import (
    JobStatus,
    PipelineStatus,
    ClipResponse,
    ClipSegment,
)
from video_maker.downloader import download_video
from video_maker.analyzer import analyze_video
from video_maker.vision import detect_faces
from video_maker.renderer import render_clip, _probe_video
from video_maker.utils import logger, cleanup_directory

# In-memory job store
_jobs: dict[str, PipelineStatus] = {}

# Thread pool for blocking I/O (one pipeline at a time on small containers)
_executor = ThreadPoolExecutor(max_workers=2)


def get_job(job_id: str) -> PipelineStatus | None:
    """Get the current status of a job."""
    return _jobs.get(job_id)


def list_jobs() -> list[PipelineStatus]:
    """List all jobs."""
    return list(_jobs.values())


def _update_job(job_id: str, **kwargs) -> None:
    """Update job status fields."""
    if job_id in _jobs:
        for k, v in kwargs.items():
            setattr(_jobs[job_id], k, v)


# ── Single-clip render worker ──────────────────────────────────────


def _render_one_clip(
    idx: int,
    segment: ClipSegment,
    video_path: Path,
    output_job_dir: Path,
    src_width: int,
    src_height: int,
    job_id: str,
) -> ClipResponse | None:
    """Render a single clip: face detection → FFmpeg render. Runs in a worker thread."""
    clip_duration = round(segment.end - segment.start, 1)
    output_filename = f"clip_{idx + 1:02d}.mp4"
    output_path = output_job_dir / output_filename
    prefix = f"[{job_id}][clip {idx + 1}]"

    try:
        # Face detection for this segment
        logger.info(f"{prefix} Detecting faces...")
        try:
            segment.face_keyframes = detect_faces(
                video_path=video_path,
                start_time=segment.start,
                end_time=segment.end,
            )
        except Exception as e:
            logger.error(f"{prefix} Face detection failed: {e}")

        # Render
        logger.info(f"{prefix} Rendering {clip_duration}s clip...")
        render_clip(
            source_path=video_path,
            segment=segment,
            output_path=output_path,
            src_width=src_width,
            src_height=src_height,
            job_id=job_id,
        )

        return ClipResponse(
            filename=output_filename,
            download_link=f"/api/clips/{job_id}/{output_filename}",
            virality_score=segment.virality_score,
            duration=clip_duration,
            title=segment.title,
            hook_reason=segment.hook_reason,
        )

    except Exception as e:
        logger.error(f"{prefix} Render failed: {e}")
        return None


# ── Main pipeline ──────────────────────────────────────────────────


def _run_pipeline_sync(job_id: str, youtube_url: str) -> None:
    """Synchronous pipeline execution (runs in thread pool)."""
    job_dir = WORKING_DIR / job_id
    output_job_dir = OUTPUT_DIR / job_id
    output_job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Download (high-res only) ────────────────────────
        _update_job(job_id, status=JobStatus.DOWNLOADING, progress="Downloading video...")
        dl_result = download_video(youtube_url, job_id)
        video_path = dl_result["video_path"]
        logger.info(f"[{job_id}] Download complete: {dl_result['title']}")

        # ── Step 2: Local analysis (scoring + Whisper + ranking) ────
        _update_job(job_id, status=JobStatus.ANALYZING, progress="Analyzing video locally (audio + visual + transcription)...")
        analysis = analyze_video(
            video_path=video_path,
            work_dir=job_dir,
            subs_path=dl_result.get("subs_path"),
        )

        if not analysis.clips:
            _update_job(
                job_id,
                status=JobStatus.FAILED,
                error="Could not identify interesting segments in this video.",
            )
            return

        num_clips = len(analysis.clips)
        for i, clip in enumerate(analysis.clips):
            logger.info(
                f"[{job_id}] Clip {i + 1}/{num_clips}: "
                f"{clip.start:.1f}s→{clip.end:.1f}s, "
                f"{len(clip.words)} subtitle words"
            )
        logger.info(f"[{job_id}] {num_clips} clips identified, starting render (workers={NUM_WORKERS})...")

        # Probe video dimensions once (not in each worker)
        probe = _probe_video(video_path)
        src_w = probe["width"]
        src_h = probe["height"]

        # ── Step 3: Render clips in parallel ────────────────────────
        _update_job(
            job_id,
            status=JobStatus.RENDERING,
            progress=f"Rendering {num_clips} clips with {NUM_WORKERS} workers...",
        )

        clip_responses = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as render_pool:
            futures = []
            for idx, segment in enumerate(analysis.clips):
                fut = render_pool.submit(
                    _render_one_clip,
                    idx,
                    segment,
                    video_path,
                    output_job_dir,
                    src_w,
                    src_h,
                    job_id,
                )
                futures.append(fut)

            for fut in futures:
                result = fut.result()
                if result is not None:
                    clip_responses.append(result)

        if not clip_responses:
            _update_job(job_id, status=JobStatus.FAILED, error="All clip renders failed.")
            return

        # ── Step 4: Cleanup temp files ──────────────────────────────
        logger.info(f"[{job_id}] Cleaning up temp files...")
        cleanup_directory(job_dir)

        # ── Done ────────────────────────────────────────────────────
        total_duration = sum(c.duration for c in clip_responses)
        _update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=f"Done! {len(clip_responses)} clips ready ({total_duration:.0f}s total).",
            clips=clip_responses,
        )
        logger.info(f"[{job_id}] Pipeline complete: {len(clip_responses)} clips rendered")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        _update_job(job_id, status=JobStatus.FAILED, error=str(e))
        cleanup_directory(job_dir)


async def start_pipeline(youtube_url: str) -> str:
    """Start the video processing pipeline in the background.

    Returns:
        job_id: Unique identifier for tracking this job.
    """
    job_id = uuid.uuid4().hex[:12]

    _jobs[job_id] = PipelineStatus(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress="Queued...",
    )

    logger.info(f"[{job_id}] Starting pipeline for: {youtube_url}")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_sync, job_id, youtube_url)

    return job_id
