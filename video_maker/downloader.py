"""Download YouTube videos using yt-dlp for analysis and rendering."""

import sys
from pathlib import Path
from video_maker.config import WORKING_DIR, PROJECT_ROOT, FFMPEG_DIR
from video_maker.utils import logger

# Add the youtube-downloader-api to sys.path so we can reuse yt-dlp
_yt_api_path = str(PROJECT_ROOT / "youtube-downloader-api")
if _yt_api_path not in sys.path:
    sys.path.insert(0, _yt_api_path)

import yt_dlp


def _make_job_dir(job_id: str) -> Path:
    """Create and return a working directory for this job."""
    job_dir = WORKING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def download_video(url: str, job_id: str) -> dict:
    """Download a YouTube video in two resolutions.

    Returns:
        dict with keys:
            - "low_res": Path to 360p video (for Gemini analysis)
            - "high_res": Path to best available video (for final render)
            - "title": Video title
            - "duration": Video duration in seconds
    """
    job_dir = _make_job_dir(job_id)
    result = {}

    # ── Step 1: Extract info ────────────────────────────────────────
    logger.info(f"[{job_id}] Extracting video info for: {url}")
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "ffmpeg_location": str(FFMPEG_DIR)}) as ydl:
        info = ydl.extract_info(url, download=False)

    result["title"] = info.get("title", "untitled")
    result["duration"] = info.get("duration", 0)
    result["width"] = info.get("width", 1920)
    result["height"] = info.get("height", 1080)

    logger.info(f"[{job_id}] Video: {result['title']} ({result['duration']}s, {result['width']}x{result['height']})")

    # ── Step 2: Download low-res for Gemini analysis ────────────────
    low_res_path = job_dir / "source_low.mp4"
    logger.info(f"[{job_id}] Downloading low-res (360p)...")
    _download_with_format(url, low_res_path, format_selector="bestvideo[height<=360]+bestaudio/best[height<=360]/worst")
    result["low_res"] = low_res_path

    # ── Step 3: Download high-res for final render ──────────────────
    high_res_path = job_dir / "source_high.mp4"
    logger.info(f"[{job_id}] Downloading high-res...")
    _download_with_format(url, high_res_path, format_selector="bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best")
    result["high_res"] = high_res_path

    logger.info(f"[{job_id}] Downloads complete.")
    return result


def _download_with_format(url: str, output_path: Path, format_selector: str) -> None:
    """Download a video with a specific format selector."""
    ydl_opts = {
        "format": format_selector,
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
