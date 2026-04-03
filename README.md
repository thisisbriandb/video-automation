# Video Maker – YouTube to Shorts Pipeline

Transform long YouTube videos (16:9) into viral short clips (9:16) optimized for TikTok/Reels.

## Features

- **AI-powered segment detection** — Gemini identifies the most viral moments
- **Dynamic face-tracking crop** — The 9:16 frame follows the speaker
- **TikTok-style subtitles** — Bold, centered, burned into the video
- **Audio normalization** — Consistent volume across clips
- **Async pipeline** — Non-blocking processing with job tracking

## Prerequisites

- **Python 3.11+**
- **FFmpeg** installed and available in PATH
- **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run the server
python -m uvicorn video_maker.app:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process` | Submit a YouTube URL for processing |
| `GET` | `/api/status/{job_id}` | Check job progress |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/clips/{job_id}/{filename}` | Download a rendered clip |

### Example Usage

```bash
# Submit a video
curl -X POST http://localhost:8001/api/process \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# Check status (use the job_id from the response)
curl http://localhost:8001/api/status/abc123def456

# Download a clip
curl -O http://localhost:8001/api/clips/abc123def456/clip_01.mp4
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `WORKING_DIR` | `./workdir` | Temporary download/processing directory |
| `OUTPUT_DIR` | `./output` | Final rendered clips directory |
| `MAX_CLIPS` | `5` | Maximum clips to extract per video |
| `CLIP_MIN_DURATION` | `15` | Minimum clip length (seconds) |
| `CLIP_MAX_DURATION` | `60` | Maximum clip length (seconds) |

## Architecture

```
video-maker/
├── youtube-downloader-api/   # Existing YouTube downloader
├── video_maker/              # Pipeline package
│   ├── app.py                # FastAPI endpoints
│   ├── config.py             # Environment configuration
│   ├── downloader.py         # yt-dlp video download
│   ├── analyzer.py           # Gemini multimodal analysis
│   ├── renderer.py           # FFmpeg crop/scale/subtitle render
│   ├── pipeline.py           # Async pipeline orchestrator
│   ├── models.py             # Pydantic data models
│   └── utils.py              # Helpers (SRT gen, cleanup)
├── .env                      # Your configuration
├── requirements.txt          # Python dependencies
└── README.md
```

## Pipeline Flow

1. **Download** — Fetch video in low-res (360p for AI analysis) and high-res (1080p for rendering)
2. **Analyze** — Upload to Gemini → detect viral segments, face positions, generate subtitles
3. **Render** — For each segment: trim → dynamic crop → scale to 1080×1920 → normalize audio → burn subtitles → export MP4
4. **Cleanup** — Remove temporary files, serve final clips
