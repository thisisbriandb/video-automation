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


# Gemini
GEMINI_API_KEY: str = _get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = _get("GEMINI_MODEL", "gemini-2.0-flash")

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
WORKING_DIR: Path = Path(_get("WORKING_DIR", str(PROJECT_ROOT / "workdir")))
OUTPUT_DIR: Path = Path(_get("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# FFmpeg
_default_ffmpeg = str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin")
FFMPEG_DIR: Path = Path(_get("FFMPEG_DIR", _default_ffmpeg))

# Pipeline
MAX_CLIPS: int = int(_get("MAX_CLIPS", "5"))
CLIP_MIN_DURATION: float = float(_get("CLIP_MIN_DURATION", "15"))
CLIP_MAX_DURATION: float = float(_get("CLIP_MAX_DURATION", "60"))

# Ensure dirs exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
