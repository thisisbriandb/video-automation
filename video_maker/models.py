"""Pydantic models for the video maker pipeline"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RenderPreset(str, Enum):
    """Visual preset for the final clip rendering."""
    DEFAULT = "default"          # 9:16 dynamic crop + Hormozi subs (existing behavior)
    PODCAST_BW = "podcast_bw"    # background removal + black & white + music


# ── Request / Response ──────────────────────────────────────────────

class ClipRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL")
    render_preset: RenderPreset = Field(
        RenderPreset.DEFAULT,
        description="Rendering preset: 'default' (TikTok subs) or 'podcast_bw' (B&W matting + music)",
    )
    music_id: Optional[str] = Field(
        None,
        description="Identifier of an audio file uploaded via /api/upload-music (only used by podcast_bw preset)",
    )


class SubtitleWord(BaseModel):
    start: float = Field(..., description="Word start time in seconds")
    end: float = Field(..., description="Word end time in seconds")
    word: str = Field(..., description="Single word text")


class SubtitleLine(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text")


class FaceKeyframe(BaseModel):
    time: float = Field(..., description="Time offset in seconds from segment start")
    x: int = Field(..., description="X-coordinate of face center in source frame (real pixels)")


class ScoredSegment(BaseModel):
    """A segment scored by local analysis (audio + visual), before Whisper."""
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    audio_score: float = Field(0.0, description="Normalized audio interest score 0-1")
    visual_score: float = Field(0.0, description="Normalized visual interest score 0-1")
    text_score: float = Field(0.0, description="Text richness score 0-1 (set after Whisper)")
    total_score: float = Field(0.0, description="Weighted combined score 0-1")


class ClipSegment(BaseModel):
    start: float = Field(..., description="Segment start time in seconds (hook timestamp)")
    end: float = Field(..., description="Segment end time in seconds")
    virality_score: int = Field(..., ge=1, le=10, description="Virality score 1-10")
    title: str = Field("", description="Short descriptive title for the clip")
    hook_reason: str = Field("", description="Why this moment is the best hook")
    face_keyframes: list[FaceKeyframe] = Field(default_factory=list)
    words: list[SubtitleWord] = Field(default_factory=list)
    language: str = Field("fr", description="Detected language code (ISO 639-1)")


class AnalysisResult(BaseModel):
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
    percent: int = 0
    video_id: str = ""
    video_title: str = ""
    error: Optional[str] = None
    clips: list[ClipResponse] = Field(default_factory=list)
    render_preset: RenderPreset = RenderPreset.DEFAULT
