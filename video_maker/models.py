"""Pydantic models for the video maker pipeline"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Request / Response ──────────────────────────────────────────────

class ClipRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL")


class SubtitleLine(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text")


class FaceKeyframe(BaseModel):
    time: float = Field(..., description="Time offset in seconds from segment start")
    x: int = Field(..., description="X-coordinate of face center in source frame (0-1920)")


class ClipSegment(BaseModel):
    start: float = Field(..., description="Segment start time in seconds (hook timestamp)")
    end: float = Field(..., description="Segment end time in seconds")
    virality_score: int = Field(..., ge=1, le=10, description="Virality score 1-10")
    title: str = Field("", description="Short descriptive title for the clip")
    hook_reason: str = Field("", description="Why this moment is the best hook")
    face_keyframes: list[FaceKeyframe] = Field(default_factory=list)
    subtitles: list[SubtitleLine] = Field(default_factory=list)


class GeminiAnalysisResult(BaseModel):
    clips: list[ClipSegment] = Field(default_factory=list)


class ClipResponse(BaseModel):
    filename: str
    download_link: str
    virality_score: int
    duration: float
    title: str = ""
    hook_reason: str = ""


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(BaseModel):
    job_id: str
    status: JobStatus
    progress: str = ""
    error: Optional[str] = None
    clips: list[ClipResponse] = Field(default_factory=list)
