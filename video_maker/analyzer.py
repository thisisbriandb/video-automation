"""Gemini multimodal analysis: detect viral segments, face positions, and subtitles."""

import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from video_maker.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CLIPS, CLIP_MIN_DURATION, CLIP_MAX_DURATION
from video_maker.models import (
    GeminiAnalysisResult,
    ClipSegment,
    FaceKeyframe,
    SubtitleLine,
)
from video_maker.utils import logger

# Initialize Gemini client
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please configure it in .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _build_analysis_prompt(duration: float, width: int, height: int) -> str:
    """Build the structured prompt for Gemini video analysis."""
    return f"""You are a world-class TikTok editor and viral content strategist.

WATCH AND UNDERSTAND THE ENTIRE VIDEO FIRST, then complete the task below.

VIDEO INFO:
- Total duration: {duration:.1f} seconds
- Resolution: {width}x{height}
- Format: Landscape (16:9)

TASK — Create ONE viral short (~60 seconds)
Your goal is to produce a single 60-second clip that maximises engagement on TikTok/Reels.

Step 1 — Understand the full content
Watch the video from start to finish. Identify the overall topic, the speaker(s), the narrative arc, the key arguments, jokes, or emotional beats.

Step 2 — Detect the HOOK
The "hook" is the single most attention-grabbing moment in the video — the sentence or scene that would make a viewer stop scrolling. It can be:
  - A shocking statement, bold claim, or controversial take
  - A punchline, joke, or unexpected reaction
  - An emotional peak or dramatic reveal
  - A concise, powerful explanation of something complex
The clip MUST START at the hook. Not before — exactly at the hook.

Step 3 — Build the clip from the hook
Starting exactly at the hook, include the next ~60 seconds of content (between {CLIP_MIN_DURATION} and {CLIP_MAX_DURATION} seconds). The clip should:
  - Open instantly with the hook (first 3 seconds grab attention)
  - Flow naturally after the hook — do NOT jump to unrelated parts
  - End on a satisfying beat (end of a sentence, a reaction, a punchline)

Step 4 — Face tracking
Provide the X-coordinate (in pixels, 0 = left, {width} = right) of the dominant speaker's face center every 5 seconds throughout the clip. If no face is visible at a given moment, use {width // 2}.

Step 5 — Subtitles
Transcribe the speech in the clip with precise timing RELATIVE TO THE CLIP START (time 0.0 = hook moment). Each subtitle line must be 1-8 words for TikTok readability.

RESPOND WITH ONLY VALID JSON (no markdown, no code fences, no explanation):
{{
  "clips": [
    {{
      "start": <hook timestamp in original video, seconds>,
      "end": <end timestamp in original video, seconds>,
      "virality_score": <1-10>,
      "title": "<catchy title, max 10 words>",
      "hook_reason": "<1 sentence explaining why this is the best hook>",
      "face_keyframes": [
        {{"time": 0.0, "x": 960}},
        {{"time": 5.0, "x": 980}}
      ],
      "subtitles": [
        {{"start": 0.0, "end": 1.5, "text": "This changes everything"}},
        {{"start": 1.5, "end": 3.2, "text": "and here is why"}}
      ]
    }}
  ]
}}"""


def _upload_video(video_path: Path) -> types.File:
    """Upload video to Gemini Files API and wait for processing."""
    client = _get_client()
    logger.info(f"Uploading video to Gemini: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

    uploaded_file = client.files.upload(file=video_path)
    logger.info(f"Upload complete. File name: {uploaded_file.name}, state: {uploaded_file.state}")

    # Wait for the file to be processed
    max_wait = 300  # 5 minutes max
    waited = 0
    while uploaded_file.state == "PROCESSING" and waited < max_wait:
        time.sleep(5)
        waited += 5
        uploaded_file = client.files.get(name=uploaded_file.name)
        logger.info(f"File state: {uploaded_file.state} (waited {waited}s)")

    if uploaded_file.state == "FAILED":
        raise RuntimeError(f"Gemini file processing failed: {uploaded_file.name}")

    return uploaded_file


def _parse_gemini_response(text: str) -> GeminiAnalysisResult:
    """Parse and validate Gemini's JSON response."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first line and last line
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)

    # Validate through Pydantic
    result = GeminiAnalysisResult(**data)

    # Filter clips by duration constraints
    valid_clips = []
    for clip in result.clips:
        duration = clip.end - clip.start
        if CLIP_MIN_DURATION <= duration <= CLIP_MAX_DURATION:
            valid_clips.append(clip)
        else:
            logger.warning(f"Skipping clip ({clip.start:.1f}-{clip.end:.1f}s): duration {duration:.1f}s out of range")

    # Keep only the single best clip (highest virality score)
    if valid_clips:
        valid_clips.sort(key=lambda c: c.virality_score, reverse=True)
        best = valid_clips[0]
        logger.info(f"Selected hook at {best.start:.1f}s: {best.hook_reason}")
        result.clips = [best]
    else:
        result.clips = []

    return result


def analyze_video(
    video_path: Path,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    max_retries: int = 2,
) -> GeminiAnalysisResult:
    """Analyze a video with Gemini to detect viral segments.

    Args:
        video_path: Path to the low-res video file
        duration: Video duration in seconds
        width: Source video width
        height: Source video height
        max_retries: Number of retries on malformed JSON

    Returns:
        GeminiAnalysisResult with list of ClipSegments
    """
    client = _get_client()

    # Upload video
    uploaded_file = _upload_video(video_path)

    # Build prompt
    prompt = _build_analysis_prompt(duration, width, height)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Sending analysis request to Gemini (attempt {attempt + 1}/{max_retries + 1})")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )

            raw_text = response.text
            logger.info(f"Gemini response received ({len(raw_text)} chars)")
            logger.debug(f"Raw response:\n{raw_text[:500]}")

            result = _parse_gemini_response(raw_text)
            logger.info(f"Analysis complete: {len(result.clips)} clips detected")
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed to parse Gemini response: {e}")
            if attempt < max_retries:
                time.sleep(2)

    # All retries exhausted
    raise RuntimeError(f"Failed to get valid analysis from Gemini after {max_retries + 1} attempts: {last_error}")


def cleanup_uploaded_file(file_name: str) -> None:
    """Delete an uploaded file from Gemini."""
    try:
        client = _get_client()
        client.files.delete(name=file_name)
        logger.info(f"Deleted uploaded file: {file_name}")
    except Exception as e:
        logger.warning(f"Failed to delete uploaded file {file_name}: {e}")
