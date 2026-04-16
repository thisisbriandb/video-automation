"""Gemini-powered video segmentation: send YouTube URL, get smart cut points."""

import json
import re
from typing import List

from video_maker.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CLIPS, CLIP_MIN_DURATION, CLIP_MAX_DURATION
from video_maker.models import ClipSegment
from video_maker.utils import logger


def _build_prompt() -> str:
    """Build the analysis prompt for Gemini."""
    return f"""Tu es un expert en montage vidéo viral pour les réseaux sociaux (TikTok, Reels, YouTube Shorts).

Analyse cette vidéo YouTube en profondeur et identifie les {MAX_CLIPS} meilleurs moments pour créer des clips viraux autonomes.

RÈGLES STRICTES :
1. Chaque clip doit durer entre {int(CLIP_MIN_DURATION)} et {int(CLIP_MAX_DURATION)} secondes
2. Chaque clip DOIT former une UNITÉ DE SENS complète — une idée, un argument, une anecdote ou une histoire ENTIÈRE
3. Le clip doit être COMPRÉHENSIBLE sans avoir vu le reste de la vidéo
4. PAS de coupure en milieu de phrase ou d'idée
5. Chaque clip doit se terminer naturellement (fin de phrase, conclusion, chute)
6. Les segments ne doivent PAS se chevaucher

HOOK (très important) :
Pour chaque clip, écris une phrase d'accroche COURTE et PERCUTANTE qui apparaîtra en texte au tout début du clip pour capter l'attention du spectateur.
Le hook doit donner envie de regarder la suite, créer de la curiosité ou de l'émotion.
Style TikTok — exemples :
- "Cet homme a rompu avec sa femme pour cette raison..."
- "La technique que personne ne connaît"
- "Il a perdu 50 000€ à cause de ça"

Retourne UNIQUEMENT un tableau JSON valide (pas de markdown, pas de ```json```, pas de texte autour) :
[
  {{
    "start": 125.0,
    "end": 210.0,
    "title": "Titre court descriptif du clip",
    "hook": "La phrase d'accroche percutante",
    "virality_score": 8
  }}
]

IMPORTANT :
- start et end sont en SECONDES (float)
- virality_score de 1 à 10
- Ordonne par virality_score décroissant
- Maximum {MAX_CLIPS} clips
- Chaque clip MINIMUM {int(CLIP_MIN_DURATION)} secondes"""


def segment_with_gemini(youtube_url: str) -> List[ClipSegment]:
    """Send YouTube URL to Gemini for intelligent video segmentation.

    Gemini watches the video directly (no upload needed) and returns
    the best segments to cut, each with a viral hook phrase.

    Returns:
        List of ClipSegment (words empty — Whisper fills them later).
    """
    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY — skipping Gemini segmentation")
        return []

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt()

    logger.info(f"Gemini ({GEMINI_MODEL}): analyzing YouTube video...")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_uri(
                    file_uri=youtube_url,
                    mime_type="video/*",
                ),
                prompt,
            ],
        )
    except Exception as e:
        logger.error(f"Gemini API failed: {e}")
        return []

    text = response.text.strip()
    logger.info(f"Gemini response received ({len(text)} chars)")

    # Parse JSON array from response
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        logger.error(f"Gemini: no JSON array found in response:\n{text[:500]}")
        return []

    try:
        segments_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.error(f"Gemini: JSON parse error: {e}\n{text[:500]}")
        return []

    clips: List[ClipSegment] = []
    for seg in segments_data:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            duration = end - start

            if duration < CLIP_MIN_DURATION * 0.8:
                logger.warning(f"Gemini: skipping short segment {start:.0f}-{end:.0f}s ({duration:.0f}s)")
                continue

            clips.append(ClipSegment(
                start=start,
                end=end,
                virality_score=max(1, min(10, int(seg.get("virality_score", 5)))),
                title=seg.get("title", ""),
                hook_reason=seg.get("hook", seg.get("hook_reason", "")),
                words=[],
                language="",
            ))
        except (KeyError, ValueError) as e:
            logger.warning(f"Gemini: invalid segment entry: {e}")
            continue

    clips.sort(key=lambda c: c.start)
    logger.info(f"Gemini: {len(clips)} segments identified:")
    for i, c in enumerate(clips):
        logger.info(
            f"  #{i+1} [{c.start:.0f}s → {c.end:.0f}s] "
            f"({c.end - c.start:.0f}s) score={c.virality_score} "
            f"hook=\"{c.hook_reason[:60]}\""
        )

    return clips
