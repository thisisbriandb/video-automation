"""FastAPI application for the Video Maker pipeline."""

import asyncio
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pathlib import Path
from video_maker.config import OUTPUT_DIR
from video_maker.models import ClipRequest, JobStatus, PipelineStatus
from video_maker.pipeline import start_pipeline, get_job, list_jobs, _job_version, MUSIC_DIR

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

    if not any(domain in url for domain in ["youtube.com", "youtu.be", "youtube-nocookie.com"]):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    job_id = await start_pipeline(
        url,
        render_preset=request.render_preset,
        music_id=request.music_id,
    )
    return get_job(job_id)


# ── Music upload (used by the podcast_bw preset) ────────────────────


_ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
_MAX_MUSIC_SIZE = 25 * 1024 * 1024  # 25 MB


@app.post("/api/upload-music")
async def upload_music(file: UploadFile = File(...)):
    """Upload a background music track for the podcast preset.

    Returns a `music_id` that should be passed to /api/process.
    """
    filename = file.filename or "track"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext or '(none)'}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTS))}",
        )

    music_id = f"{uuid.uuid4().hex[:12]}{ext}"
    target = MUSIC_DIR / music_id

    size = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_MUSIC_SIZE:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {_MAX_MUSIC_SIZE // 1024 // 1024} MB)",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    finally:
        await file.close()

    return JSONResponse(
        content={
            "music_id": music_id,
            "filename": filename,
            "size_bytes": size,
        }
    )


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
