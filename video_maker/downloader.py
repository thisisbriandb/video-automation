"""Download YouTube videos using yt-dlp for analysis and rendering."""

import base64
import os
import sys
import tempfile
from pathlib import Path
from video_maker.config import WORKING_DIR, PROJECT_ROOT, FFMPEG_DIR
from video_maker.utils import logger


def _get_cookies_path() -> str | None:
    """Return path to a Netscape cookies file, if available.

    Checks in order:
      1. YOUTUBE_COOKIES_FILE env var (direct path)
      2. YOUTUBE_COOKIES env var (base64-encoded cookies.txt content)
    """
    # Direct file path
    cfile = os.environ.get("YOUTUBE_COOKIES_FILE", "")
    if cfile and Path(cfile).is_file():
        return cfile

    # Base64-encoded content → write to temp file
    b64 = os.environ.get("YOUTUBE_COOKIES", "")
    if b64:
        try:
            # Fix padding if truncated/corrupted during paste
            b64 = b64.strip()
            missing_pad = len(b64) % 4
            if missing_pad:
                b64 += "=" * (4 - missing_pad)
            raw = base64.b64decode(b64)
            tmp = Path(tempfile.gettempdir()) / "yt_cookies.txt"
            tmp.write_bytes(raw)
            logger.info(f"YouTube cookies written to {tmp} ({len(raw)} bytes)")
            return str(tmp)
        except Exception as e:
            logger.warning(f"Failed to decode YOUTUBE_COOKIES: {e}")

    return None

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
    """Download a YouTube video (high-res only).

    Returns:
        dict with keys:
            - "video_path": Path to best available video (for analysis + render)
            - "title": Video title
            - "duration": Video duration in seconds
            - "width": Video width
            - "height": Video height
    """
    job_dir = _make_job_dir(job_id)
    result = {}

    # ── Step 1: Extract info ────────────────────────────────────────
    logger.info(f"[{job_id}] Extracting video info for: {url}")
    info_opts: dict = {"quiet": True, "no_warnings": True, "ffmpeg_location": str(FFMPEG_DIR)}
    cookies = _get_cookies_path()
    if cookies:
        info_opts["cookiefile"] = cookies
        logger.info("Using YouTube cookies for authentication")
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    result["title"] = info.get("title", "untitled")
    result["duration"] = info.get("duration", 0)
    result["width"] = info.get("width", 1920)
    result["height"] = info.get("height", 1080)

    logger.info(f"[{job_id}] Video: {result['title']} ({result['duration']}s, {result['width']}x{result['height']})")

    # ── Step 2: Download high-res (1080p, fallback 720p) ─────────────
    video_path = job_dir / "source.mp4"
    logger.info(f"[{job_id}] Downloading (1080p target)...")
    try:
        _download_with_format(
            url, video_path,
            format_selector=(
                "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
            ),
        )
    except Exception as e:
        logger.warning(f"[{job_id}] 1080p download failed ({e}), retrying at 720p...")
        _download_with_format(
            url, video_path,
            format_selector=(
                "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height<=720]+bestaudio/best[height<=720]"
            ),
        )

    result["video_path"] = video_path
    size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"[{job_id}] Download complete: {size_mb:.1f} MB")
    return result


def _dl_progress_hook(d: dict) -> None:
    """Log download progress at meaningful intervals."""
    if d.get("status") == "downloading":
        pct = d.get("_percent_str", "?").strip()
        speed = d.get("_speed_str", "?").strip()
        eta = d.get("_eta_str", "?").strip()
        # Log every ~10%
        try:
            pct_val = float(d.get("downloaded_bytes", 0)) / max(float(d.get("total_bytes", 1)), 1) * 100
            if int(pct_val) % 10 < 2:  # roughly every 10%
                logger.info(f"Download: {pct} at {speed}, ETA {eta}")
        except (ValueError, TypeError):
            pass
    elif d.get("status") == "finished":
        logger.info(f"Download finished, merging/converting...")


def _download_with_format(url: str, output_path: Path, format_selector: str) -> None:
    """Download a video with a specific format selector."""
    ydl_opts = {
        "format": format_selector,
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
        "overwrites": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        # Retry & timeout settings
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "http_chunk_size": 10485760,  # 10 MB chunks
        "progress_hooks": [_dl_progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }
    cookies = _get_cookies_path()
    if cookies:
        ydl_opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
