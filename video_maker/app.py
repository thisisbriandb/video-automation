"""FastAPI application for the Video Maker pipeline."""

import asyncio

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pathlib import Path
from video_maker.config import OUTPUT_DIR
from video_maker.models import ClipRequest, JobStatus, PipelineStatus
from video_maker.pipeline import start_pipeline, get_job, list_jobs, _job_version

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Video Maker – YouTube to Shorts",
    version="0.1.0",
    description="Transform long YouTube videos into viral 9:16 short clips with AI-powered segment detection, dynamic cropping, and TikTok-style subtitles.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/api/process", response_model=PipelineStatus, status_code=status.HTTP_202_ACCEPTED)
async def process_video(request: ClipRequest):
    """Submit a YouTube URL for processing.

    Returns a job_id to track progress via GET /api/status/{job_id}.
    """
    url = request.youtube_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="youtube_url is required")

    # Basic URL validation
    if not any(domain in url for domain in ["youtube.com", "youtu.be", "youtube-nocookie.com"]):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    job_id = await start_pipeline(url)
    return get_job(job_id)


@app.get("/api/status/{job_id}", response_model=PipelineStatus)
async def get_status(job_id: str):
    """Get the current status of a processing job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/api/jobs", response_model=list[PipelineStatus])
async def get_all_jobs():
    """List all processing jobs."""
    return list_jobs()


@app.get("/api/events/{job_id}")
async def stream_events(job_id: str):
    """SSE endpoint for real-time job progress updates."""
    async def generate():
        last_version = -1
        while True:
            current = _job_version.get(job_id, 0)
            if current != last_version:
                last_version = current
                job = get_job(job_id)
                if job:
                    yield f"data: {job.model_dump_json()}\n\n"
                    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                        break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/clips/{job_id}/{filename}")
async def get_clip(job_id: str, filename: str):
    """Download a rendered clip file."""
    # Sanitize filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    clip_path = OUTPUT_DIR / job_id / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename,
    )
