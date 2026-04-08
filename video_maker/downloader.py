"""Download YouTube videos using yt-dlp for analysis and rendering."""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from video_maker.config import WORKING_DIR, PROJECT_ROOT, FFMPEG_DIR
from video_maker.utils import logger


def _get_cookies_path() -> str | None:
    """Return path to a Netscape cookies file, if available.

    Checks in order:
      1. YOUTUBE_COOKIES_FILE env var (direct path to existing file)
      2. YOUTUBE_COOKIES env var (raw OR base64-encoded cookies.txt content)
    """
    # Direct file path
    cfile = os.environ.get("YOUTUBE_COOKIES_FILE", "")
    if cfile and Path(cfile).is_file():
        # Normalize CRLF → LF (Windows cookies on Linux Docker)
        content = Path(cfile).read_bytes()
        if b"\r\n" in content:
            content = content.replace(b"\r\n", b"\n")
            Path(cfile).write_bytes(content)
            logger.info(f"Normalized CRLF in cookies file: {cfile}")
        newline_count = content.count(b"\n")
        logger.info(f"Cookies file: {cfile} ({len(content)} bytes, {newline_count} lines)")
        return cfile

    raw_env = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not raw_env:
        return None

    tmp = Path(tempfile.gettempdir()) / "yt_cookies.txt"

    # If it looks like Netscape cookie content, write directly
    if raw_env.startswith("# Netscape") or raw_env.startswith("# HTTP") or "\t" in raw_env[:200]:
        tmp.write_text(raw_env, encoding="utf-8")
        logger.info(f"YouTube cookies (raw) written to {tmp} ({len(raw_env)} chars)")
        return str(tmp)

    # Otherwise try base64 decode
    try:
        cleaned = raw_env.replace("\n", "").replace("\r", "").replace(" ", "")
        missing_pad = len(cleaned) % 4
        if missing_pad:
            cleaned += "=" * (4 - missing_pad)
        raw = base64.b64decode(cleaned)
        tmp.write_bytes(raw)
        logger.info(f"YouTube cookies (b64) written to {tmp} ({len(raw)} bytes)")
        return str(tmp)
    except Exception as e:
        logger.warning(f"Failed to decode YOUTUBE_COOKIES: {e}")

    return None

# Add the youtube-downloader-api to sys.path so we can reuse yt-dlp
_yt_api_path = str(PROJECT_ROOT / "youtube-downloader-api")
if _yt_api_path not in sys.path:
    sys.path.insert(0, _yt_api_path)

import yt_dlp

# Log Node.js availability at import time (critical for yt-dlp JS challenges)
_node_path = shutil.which("node")
if _node_path:
    try:
        _node_ver = subprocess.check_output([_node_path, "--version"], text=True).strip()
        logger.info(f"Node.js found: {_node_path} ({_node_ver})")
    except Exception:
        logger.warning(f"Node.js binary found at {_node_path} but failed to run")
else:
    logger.warning("Node.js NOT found in PATH — yt-dlp JS challenges will fail!")


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

    # ── Step 1: Extract info (skip format processing) ─────────────
    logger.info(f"[{job_id}] Extracting video info for: {url}")
    info_opts: dict = {
        "quiet": True, "no_warnings": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    }
    cookies = _get_cookies_path()
    if cookies:
        info_opts["cookiefile"] = cookies
        logger.info("Using YouTube cookies for authentication")
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False, process=False)

    # Log available formats for debugging
    formats = info.get("formats") or []
    logger.info(f"[{job_id}] YouTube returned {len(formats)} formats")
    if formats:
        sample = [(f.get('format_id'), f.get('ext'), f.get('height')) for f in formats[:5]]
        logger.info(f"[{job_id}] Sample formats: {sample}")
    else:
        logger.warning(f"[{job_id}] No formats returned! Video may be restricted.")

    result["title"] = info.get("title", "untitled")
    result["duration"] = info.get("duration", 0)
    result["width"] = info.get("width", 1920)
    result["height"] = info.get("height", 1080)

    logger.info(f"[{job_id}] Video: {result['title']} ({result['duration']}s, {result['width']}x{result['height']})")

    # ── Step 2: Download high-res (best available, prefer ≤1080p) ────
    video_path = job_dir / "source.mp4"
    logger.info(f"[{job_id}] Downloading (best available)...")
    _download_with_format(
        url, video_path,
        format_selector=(
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1080]+bestaudio/"
            "bestvideo+bestaudio/best"
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
        "extractor_args": {"youtube": {"player_client": ["web"]}},
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
