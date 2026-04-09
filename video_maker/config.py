"""Configuration loaded from .env file"""

import os
from pathlib import Path
from dotenv import dotenv_values

# Load from .env in project root
_env_path = Path(__file__).parent.parent / ".env"
_env = dotenv_values(_env_path)

# Also allow real environment variables to override .env
def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, _env.get(key, default))


# Directories
PROJECT_ROOT = Path(__file__).parent.parent
WORKING_DIR: Path = Path(_get("WORKING_DIR", str(PROJECT_ROOT / "workdir")))
OUTPUT_DIR: Path = Path(_get("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# FFmpeg – auto-detect: env var > Windows default > Linux system
def _default_ffmpeg_dir() -> str:
    win_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin"
    if win_path.exists():
        return str(win_path)
    # Linux / Docker: ffmpeg is on PATH
    return "/usr/bin"

FFMPEG_DIR: Path = Path(_get("FFMPEG_DIR", _default_ffmpeg_dir()))

import multiprocessing

# Pipeline
MAX_CLIPS: int = int(_get("MAX_CLIPS", "10"))
CLIP_MIN_DURATION: float = float(_get("CLIP_MIN_DURATION", "60"))
CLIP_MAX_DURATION: float = float(_get("CLIP_MAX_DURATION", "90"))

# Scoring
SCORING_WINDOW: float = float(_get("SCORING_WINDOW", "30"))  # seconds per analysis window
SCORING_HOP: float = float(_get("SCORING_HOP", "10"))  # overlap hop between windows
TOP_PRESCORE: int = int(_get("TOP_PRESCORE", "40"))  # candidate segments before merge+expand

# Worker count: default 2 for cloud containers (Railway etc.) where cpu_count()
# returns the host's vCPUs (8-16) not the allocated amount (1-2).
# Override via NUM_WORKERS env var for beefier machines.
NUM_WORKERS: int = int(_get("NUM_WORKERS", "2"))

# Subtitles
SUBTITLE_WORDS_PER_CHUNK: int = int(_get("SUBTITLE_WORDS_PER_CHUNK", "3"))
SUBTITLE_UPPERCASE: bool = _get("SUBTITLE_UPPERCASE", "true").lower() == "true"

# Whisper
WHISPER_MODEL: str = _get("WHISPER_MODEL", "small")

# Ensure dirs exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
