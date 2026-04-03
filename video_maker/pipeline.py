"""Pipeline orchestrator: download → analyze → render, with async job tracking."""

import asyncio
import uuid
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from video_maker.config import WORKING_DIR, OUTPUT_DIR
from video_maker.models import (
    JobStatus,
    PipelineStatus,
    ClipResponse,
    ClipSegment,
)
from video_maker.downloader import download_video
from video_maker.analyzer import analyze_video
from video_maker.renderer import render_clip
from video_maker.utils import logger, cleanup_directory

# In-memory job store
_jobs: dict[str, PipelineStatus] = {}

# Thread pool for blocking I/O (download, Gemini, FFmpeg)
_executor = ThreadPoolExecutor(max_workers=4)


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


def _run_pipeline_sync(job_id: str, youtube_url: str) -> None:
    """Synchronous pipeline execution (runs in thread pool)."""
    job_dir = WORKING_DIR / job_id
    output_job_dir = OUTPUT_DIR / job_id
    output_job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Download ────────────────────────────────────────
        _update_job(job_id, status=JobStatus.DOWNLOADING, progress="Downloading video...")
        dl_result = download_video(youtube_url, job_id)
        logger.info(f"[{job_id}] Download complete: {dl_result['title']}")

        # ── Step 2: Analyze with Gemini ─────────────────────────────
        _update_job(job_id, status=JobStatus.ANALYZING, progress="Analyzing video with AI...")
        analysis = analyze_video(
            video_path=dl_result["low_res"],
            duration=dl_result["duration"],
            width=dl_result.get("width", 1920),
            height=dl_result.get("height", 1080),
        )

        if not analysis.clips:
            _update_job(
                job_id,
                status=JobStatus.FAILED,
                error="Gemini could not identify a hook in this video.",
            )
            return

        # Take the single best clip (first one returned by Gemini)
        segment = analysis.clips[0]
        clip_duration = round(segment.end - segment.start, 1)
        logger.info(
            f"[{job_id}] Hook detected at {segment.start:.1f}s "
            f"(score {segment.virality_score}/10, {clip_duration}s): "
            f"{segment.hook_reason}"
        )

        # ── Step 3: Render the clip ─────────────────────────────────
        _update_job(job_id, status=JobStatus.RENDERING, progress="Rendering clip from hook...")

        output_filename = "clip.mp4"
        output_path = output_job_dir / output_filename

        try:
            render_clip(
                source_path=dl_result["high_res"],
                segment=segment,
                output_path=output_path,
                src_width=dl_result.get("width", 1920),
                src_height=dl_result.get("height", 1080),
                job_id=job_id,
            )
        except Exception as e:
            logger.error(f"[{job_id}] Failed to render clip: {e}")
            _update_job(
                job_id,
                status=JobStatus.FAILED,
                error=f"Render failed: {e}",
            )
            return

        clip_response = ClipResponse(
            filename=output_filename,
            download_link=f"/api/clips/{job_id}/{output_filename}",
            virality_score=segment.virality_score,
            duration=clip_duration,
            title=segment.title,
            hook_reason=segment.hook_reason,
        )

        # ── Step 4: Cleanup temp files ──────────────────────────────
        logger.info(f"[{job_id}] Cleaning up temp files...")
        cleanup_directory(job_dir)

        # ── Done ────────────────────────────────────────────────────
        _update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=f"Done! 1 clip ready ({clip_duration}s).",
            clips=[clip_response],
        )
        logger.info(f"[{job_id}] Pipeline complete: {output_filename} ({clip_duration}s)")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        _update_job(job_id, status=JobStatus.FAILED, error=str(e))
        # Attempt cleanup even on failure
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

    # Run the blocking pipeline in the thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_sync, job_id, youtube_url)

    return job_id
