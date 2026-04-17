"""Gemini-powered video segmentation: send YouTube URL, get smart cut points."""

import json
import re
from typing import List

from video_maker.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CLIPS, CLIP_MIN_DURATION
from video_maker.models import ClipSegment
from video_maker.utils import logger


def _build_prompt() -> str:
    """Build the analysis prompt for Gemini."""
    return f"""Tu es un expert en montage vidéo viral pour les réseaux sociaux (TikTok, Reels, YouTube Shorts).

Analyse cette vidéo YouTube et trouve les {MAX_CLIPS} meilleurs POINTS DE DÉPART pour des clips viraux.

⚠️ TU NE CHOISIS QUE LE POINT DE DÉPART (start). La durée du clip sera automatiquement de {int(CLIP_MIN_DURATION)} secondes après "start".

RÈGLE CRITIQUE — AUTONOMIE DU CLIP :
- Le spectateur n'a PAS vu le reste de la vidéo
- À "start", le sujet DOIT être compréhensible IMMÉDIATEMENT sans aucun contexte préalable
- Le speaker doit être en train de COMMENCER une nouvelle idée, anecdote, ou argument
- JAMAIS de "start" en milieu de phrase, milieu de raisonnement, ou qui fait référence à quelque chose dit avant
- Cherche les moments où le speaker dit "En fait...", "Le truc c'est que...", "J'ai une anecdote...", "Le problème avec...", ou commence un nouveau sujet
- Si l'intervenant parle d'un sujet depuis 2 minutes, le "start" doit être au DÉBUT de ce sujet, pas au milieu
- Les {int(CLIP_MIN_DURATION)} secondes APRÈS "start" doivent former un contenu cohérent et intéressant
- Les segments ne doivent PAS se chevaucher (minimum {int(CLIP_MIN_DURATION)}s entre deux "start")

HOOK (très important) :
Le hook est une accroche très courte (MAXIMUM 6-7 mots) à écrire sur la vidéo pour capter l'attention immédiatement.
Ce n'est PAS un résumé. C'est une PUNCHLINE style titre TikTok / accroche publicitaire.
Exemples de BONS hooks :
- "Le risque de ne rien lancer"
- "Personne ne vous dit ça"
- "50 000€ perdus à cause de ça"
- "La méthode interdite"
- "Il a tout quitté pour ça"
Exemples de MAUVAIS hooks (trop longs, trop descriptifs) :
- "Il explique pourquoi il faut se lancer dans l'entrepreneuriat" ❌
- "La technique que personne ne connaît pour gagner de l'argent" ❌

Retourne UNIQUEMENT un tableau JSON valide (pas de markdown, pas de ```json```, pas de texte autour) :
[
  {{
    "start": 2120.0,
    "title": "Le coût de l'inaction",
    "hook": "Le risque de ne rien lancer",
    "virality_score": 9
  }}
]

IMPORTANT :
- start est en SECONDES (float). Exemple : 35 minutes et 20 secondes = 2120.0
- NE PAS retourner de "end" — la durée est fixée automatiquement à {int(CLIP_MIN_DURATION)}s
- virality_score de 1 à 10
- Ordonne par virality_score décroissant
- Exactement {MAX_CLIPS} clips
- Hook : MAXIMUM 7 mots, style punchline"""


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
            # Force exact duration — Gemini only picks start points
            end = start + CLIP_MIN_DURATION

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
