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
        # Log first non-comment cookie domain for debugging expiry
        for line in content.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 5:
                    domain = parts[0]
                    expiry = parts[4]
                    logger.info(f"  Cookie sample: domain={domain}, expires={expiry}")
                    break
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

def _get_auth_opts() -> dict:
    """Return yt-dlp auth options (cookies + proxy if available)."""
    opts = {}
    cookies = _get_cookies_path()
    if cookies:
        opts["cookiefile"] = cookies
    proxy = os.environ.get("YOUTUBE_PROXY", "").strip()
    if proxy:
        opts["proxy"] = proxy
    return opts


# Log versions & Node.js availability at import time
logger.info(f"yt-dlp version: {yt_dlp.version.__version__}")
logger.info(f"Auth: {'cookies' if _get_cookies_path() else 'NONE (will likely fail)'}")
_proxy = os.environ.get("YOUTUBE_PROXY", "").strip()
if _proxy:
    logger.info(f"Proxy: {_proxy.split('@')[-1] if '@' in _proxy else _proxy}")
else:
    logger.info("Proxy: NONE (datacenter IP — YouTube may block)")
try:
    import yt_dlp_ejs
    _ejs_ver = getattr(yt_dlp_ejs, "__version__", getattr(yt_dlp_ejs, "version", "unknown"))
    logger.info(f"yt-dlp-ejs installed: {_ejs_ver}")
except ImportError:
    logger.warning("yt-dlp-ejs NOT installed — EJS challenge solver scripts missing!")

for _rt_name in ("deno", "node", "bun"):
    _rt_path = shutil.which(_rt_name)
    if _rt_path:
        try:
            _rt_ver = subprocess.check_output([_rt_path, "--version"], text=True).strip().split("\n")[0]
            logger.info(f"{_rt_name} found: {_rt_path} ({_rt_ver})")
        except Exception:
            logger.warning(f"{_rt_name} binary found at {_rt_path} but failed to run")
    else:
        logger.info(f"{_rt_name}: not installed")


def _make_job_dir(job_id: str) -> Path:
    """Create and return a working directory for this job."""
    job_dir = WORKING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _download_subtitles(url: str, job_dir: Path) -> Path | None:
    """Download YouTube auto-generated subtitles in JSON3 format (word-level timestamps).

    Returns path to the JSON3 file, or None if unavailable.
    """
    subs_template = str(job_dir / "subs")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": ["fr", "en"],
        "skip_download": True,
        "outtmpl": subs_template,
        "ffmpeg_location": str(FFMPEG_DIR),
        **_get_auth_opts(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.warning(f"Subtitle download failed: {e}")
        return None

    # Prefer fr > en > any available
    for lang in ["fr", "en"]:
        p = job_dir / f"subs.{lang}.json3"
        if p.exists() and p.stat().st_size > 0:
            logger.info(f"YouTube subtitles downloaded: {p.name} ({p.stat().st_size:,} bytes)")
            return p

    json3_files = sorted(job_dir.glob("subs.*.json3"))
    if json3_files:
        p = json3_files[0]
        logger.info(f"YouTube subtitles downloaded: {p.name} ({p.stat().st_size:,} bytes)")
        return p

    logger.info("No YouTube auto-generated subtitles available")
    return None


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
        **_get_auth_opts(),
    }
    logger.info(f"Auth: {'cookies' if 'cookiefile' in info_opts else 'none'}")
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
            # Prefer H.264 (avc1) — AV1 software decoding is ~5-10x slower on CPU
            "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
            "bestvideo[height<=1080][vcodec^=avc1]+bestaudio/"
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1080]+bestaudio/"
            "bestvideo+bestaudio/best"
        ),
    )

    result["video_path"] = video_path
    size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"[{job_id}] Download complete: {size_mb:.1f} MB")

    # ── Step 3: Download YouTube subtitles (word-level JSON3) ──────
    logger.info(f"[{job_id}] Fetching YouTube subtitles...")
    subs_path = _download_subtitles(url, job_dir)
    result["subs_path"] = subs_path

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
        "verbose": True,
        "noprogress": False,
        "overwrites": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        # Retry & timeout settings
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "socket_timeout": 30,
        "http_chunk_size": 10485760,  # 10 MB chunks
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        "sleep_interval_requests": 1,
        "progress_hooks": [_dl_progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }
    ydl_opts.update(_get_auth_opts())
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
