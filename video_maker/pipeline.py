"""Pipeline orchestrator: download → local scoring → whisper → render (parallel)."""

import asyncio
import re
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from video_maker.config import WORKING_DIR, OUTPUT_DIR, NUM_WORKERS, RENDER_WORKERS
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
_job_version: dict[str, int] = {}  # incremented on each update, for SSE

# Thread pool for blocking I/O (one pipeline at a time on small containers)
_executor = ThreadPoolExecutor(max_workers=2)


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    m = re.search(r'(?:v=|/v/|youtu\.be/|/shorts/)([a-zA-Z0-9_-]{11})', url)
    return m.group(1) if m else ""


def get_job(job_id: str) -> PipelineStatus | None:
    """Get the current status of a job."""
    return _jobs.get(job_id)


def list_jobs() -> list[PipelineStatus]:
    """List all jobs."""
    return list(_jobs.values())


def _update_job(job_id: str, **kwargs) -> None:
    """Update job status fields and bump version for SSE."""
    if job_id in _jobs:
        for k, v in kwargs.items():
            setattr(_jobs[job_id], k, v)
        _job_version[job_id] = _job_version.get(job_id, 0) + 1


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
    """Render a single clip: face detection → (dubbing) → FFmpeg render. Runs in a worker thread."""
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

        # Dubbing: translate + TTS if source is not French
        dubbed_audio_path = None
        if segment.language and segment.language != "fr" and segment.words:
            logger.info(f"{prefix} Dubbing: {segment.language} → fr")
            try:
                from video_maker.dubbing import create_dubbed_audio, translate_words
                dubbed_audio_path = create_dubbed_audio(
                    words=segment.words,
                    source_lang=segment.language,
                    clip_start=segment.start,
                    clip_duration=clip_duration,
                    source_path=video_path,
                    work_dir=output_job_dir,
                )
                # Replace subtitles with French translation
                segment.words = translate_words(segment.words, segment.language)
                logger.info(f"{prefix} Dubbing complete: audio={'OK' if dubbed_audio_path else 'FAILED'}")
            except Exception as e:
                logger.error(f"{prefix} Dubbing failed: {e}")

        # Render
        logger.info(f"{prefix} Rendering {clip_duration}s clip...")
        render_clip(
            source_path=video_path,
            segment=segment,
            output_path=output_path,
            src_width=src_width,
            src_height=src_height,
            job_id=job_id,
            dubbed_audio_path=dubbed_audio_path,
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


def _cleanup_previous_runs() -> None:
    """Wipe all previous workdir/output data to free disk space before a new run."""
    for d in (WORKING_DIR, OUTPUT_DIR):
        if d.exists():
            for child in d.iterdir():
                if child.is_dir():
                    cleanup_directory(child)
                else:
                    child.unlink(missing_ok=True)
    logger.info(f"Disk cleanup done (workdir + output cleared)")


def _run_pipeline_sync(job_id: str, youtube_url: str) -> None:
    """Synchronous pipeline execution (runs in thread pool)."""
    # Free disk space from previous runs before downloading
    _cleanup_previous_runs()

    job_dir = WORKING_DIR / job_id
    output_job_dir = OUTPUT_DIR / job_id
    output_job_dir.mkdir(parents=True, exist_ok=True)

    video_id = _extract_video_id(youtube_url)

    try:
        # ── Step 1: Download (high-res only) ────────────────────────
        def _on_dl_progress(pct):
            _update_job(
                job_id,
                percent=min(30, int(pct * 0.30)),
                progress=f"T\u00e9l\u00e9chargement... {int(pct)}%",
            )

        _update_job(
            job_id,
            status=JobStatus.DOWNLOADING,
            progress="T\u00e9l\u00e9chargement...",
            percent=1,
            video_id=video_id,
        )
        dl_result = download_video(youtube_url, job_id, progress_callback=_on_dl_progress)
        video_path = dl_result["video_path"]
        _update_job(
            job_id,
            percent=32,
            progress="T\u00e9l\u00e9chargement termin\u00e9",
            video_title=dl_result.get("title", ""),
        )
        logger.info(f"[{job_id}] Download complete: {dl_result['title']}")

        # ── Step 2: Analysis (Gemini or local scoring + Whisper) ────
        from video_maker.config import GEMINI_API_KEY
        analysis_mode = "Gemini" if GEMINI_API_KEY else "locale"
        _update_job(
            job_id,
            status=JobStatus.ANALYZING,
            progress=f"Analyse {analysis_mode}...",
            percent=35,
        )
        analysis = analyze_video(
            video_path=video_path,
            work_dir=job_dir,
            youtube_url=youtube_url,
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
                f"{clip.start:.1f}s\u2192{clip.end:.1f}s, "
                f"{len(clip.words)} subtitle words"
            )
        logger.info(f"[{job_id}] {num_clips} clips identified, starting render (workers={RENDER_WORKERS})...")

        # Probe video dimensions once (not in each worker)
        probe = _probe_video(video_path)
        src_w = probe["width"]
        src_h = probe["height"]

        # ── Step 3: Render clips in parallel ────────────────────
        _update_job(
            job_id,
            status=JobStatus.RENDERING,
            progress=f"Rendu clip 0/{num_clips}...",
            percent=65,
        )

        clip_responses = []
        rendered_count = [0]  # mutable for closure

        def _on_clip_done(fut):
            res = fut.result()
            if res is not None:
                clip_responses.append(res)
            rendered_count[0] += 1
            pct = 65 + int(30 * rendered_count[0] / num_clips)
            _update_job(
                job_id,
                percent=pct,
                progress=f"Rendu clip {rendered_count[0]}/{num_clips}...",
                clips=list(clip_responses),
            )

        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as render_pool:
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
                fut.add_done_callback(_on_clip_done)
                futures.append(fut)

            # Wait for all
            for fut in futures:
                fut.result()

        failed_count = num_clips - len(clip_responses)

        if not clip_responses:
            _update_job(job_id, status=JobStatus.FAILED, error=f"Les {num_clips} rendus ont échoué.")
            return

        # ── Step 4: Cleanup temp files ──────────────────────────
        _update_job(job_id, percent=97, progress="Nettoyage...")
        logger.info(f"[{job_id}] Cleaning up temp files...")
        cleanup_directory(job_dir)

        # ── Done ────────────────────────────────────────
        total_duration = sum(c.duration for c in clip_responses)
        if failed_count > 0:
            msg = (
                f"Terminé avec erreurs : {len(clip_responses)}/{num_clips} clips OK "
                f"({total_duration:.0f}s), {failed_count} échoué(s)"
            )
            logger.warning(f"[{job_id}] {msg}")
        else:
            msg = f"Terminé ! {len(clip_responses)} clips ({total_duration:.0f}s au total)"
        _update_job(
            job_id,
            status=JobStatus.COMPLETED,
            percent=100,
            progress=msg,
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
