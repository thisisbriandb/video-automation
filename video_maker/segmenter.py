"""Gemini-powered video segmentation: send YouTube URL, get smart cut points."""

import json
import re
import time
from typing import List

from video_maker.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_YOUTUBE_SECONDS,
    GEMINI_SEGMENT_RETRIES,
    MAX_CLIPS,
    CLIP_MIN_DURATION,
    CLIP_MAX_DURATION,
)
from video_maker.models import ClipSegment
from video_maker.utils import logger


# Ask Gemini for more clips than needed — some will be filtered out
_GEMINI_REQUEST_CLIPS = 10


def _build_prompt(video_duration_s: float = 0) -> str:
    """Build the analysis prompt for Gemini."""
    min_dur = int(CLIP_MIN_DURATION)
    max_dur = int(CLIP_MAX_DURATION)
    duration_hint = ""
    if video_duration_s > 0:
        mins = int(video_duration_s // 60)
        secs = int(video_duration_s % 60)
        max_end = video_duration_s
        duration_hint = (
            f"\n⚠️ DURÉE DE LA VIDÉO : {video_duration_s:.0f} secondes ({mins}m{secs:02d}s).\n"
            f"- INTERDIT de retourner un \"end\" supérieur à {max_end:.0f}s\n"
            f"- Tous les timestamps (start et end) doivent être entre 0 et {max_end:.0f}s\n"
        )

    return f"""Tu es un expert en montage vidéo viral pour les réseaux sociaux (TikTok, Reels, YouTube Shorts).

Analyse cette vidéo YouTube et trouve les {_GEMINI_REQUEST_CLIPS} meilleurs MOMENTS à découper en clips viraux.
{duration_hint}
RÈGLE CRITIQUE — DURÉE :
- Chaque clip doit durer ENTRE {min_dur} et {max_dur} secondes
- MINIMUM {min_dur}s (non négociable — pas de clip plus court)
- MAXIMUM {max_dur}s (non négociable — pas de clip plus long)
- Dans cette fourchette, choisis la durée qui permet de TERMINER proprement l'explication/l'anecdote
- Il vaut mieux un clip de {max_dur}s qui finit sur une punchline qu'un clip de {min_dur}s coupé en pleine phrase

RÈGLE CRITIQUE — AUTONOMIE DU CLIP (start) :
- Le spectateur n'a PAS vu le reste de la vidéo
- À "start", le sujet DOIT être compréhensible IMMÉDIATEMENT sans aucun contexte préalable
- Le speaker doit être en train de COMMENCER une nouvelle idée, anecdote, ou argument
- JAMAIS de "start" en milieu de phrase, milieu de raisonnement, ou qui fait référence à quelque chose dit avant
- Cherche les moments où le speaker dit "En fait...", "Le truc c'est que...", "J'ai une anecdote...", "Le problème avec...", ou commence un nouveau sujet
- Si l'intervenant parle d'un sujet depuis 2 minutes, le "start" doit être au DÉBUT de ce sujet, pas au milieu

RÈGLE CRITIQUE — FIN NATURELLE (end) :
- "end" DOIT tomber sur une fin de phrase complète, une conclusion, une punchline, ou une transition naturelle
- JAMAIS de "end" en plein milieu d'une explication, d'un exemple, ou d'une phrase inachevée
- Si une explication nécessite {max_dur}s pour être complète, utilise les {max_dur}s complets
- Si l'idée se boucle proprement à {min_dur+10}s, coupe là
- Les segments ne doivent PAS se chevaucher

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
    "end": 2195.0,
    "title": "Le coût de l'inaction",
    "hook": "Le risque de ne rien lancer",
    "virality_score": 9
  }}
]

IMPORTANT :
- start et end sont en SECONDES (float). Exemple : 35 minutes et 20 secondes = 2120.0
- end - start DOIT être entre {min_dur} et {max_dur} secondes
- virality_score de 1 à 10
- Ordonne par virality_score décroissant
- Exactement {_GEMINI_REQUEST_CLIPS} clips
- Hook : MAXIMUM 7 mots, style punchline"""


def segment_with_gemini(youtube_url: str, video_duration_s: float = 0) -> List[ClipSegment]:
    """Send YouTube URL to Gemini for intelligent video segmentation.

    Gemini watches the video directly (no upload needed) and returns
    the best segments to cut, each with a viral hook phrase.

    Returns:
        List of ClipSegment (words empty — Whisper fills them later).
    """
    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY — skipping Gemini segmentation")
        return []

    if GEMINI_MAX_YOUTUBE_SECONDS > 0 and video_duration_s > GEMINI_MAX_YOUTUBE_SECONDS:
        logger.warning(
            f"Gemini segmentation ignorée : durée vidéo {video_duration_s:.0f}s "
            f"dépasse GEMINI_MAX_YOUTUBE_SECONDS={GEMINI_MAX_YOUTUBE_SECONDS:.0f}s "
            f"(réf. ~90 min pour YouTube dans la doc Gemini). "
            f"Tu peux monter GEMINI_MAX_YOUTUBE_SECONDS dans .env ou t’appuyer sur le scoring local."
        )
        return []

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(video_duration_s=video_duration_s)
    if video_duration_s > 0:
        logger.info(f"Gemini: video duration hint = {video_duration_s:.0f}s, requesting {_GEMINI_REQUEST_CLIPS} clips")

    logger.info(f"Gemini ({GEMINI_MODEL}): analyzing YouTube video...")

    response = None
    last_err: Exception | None = None
    for attempt in range(GEMINI_SEGMENT_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_uri(
                        file_uri=youtube_url,
                        # Vertex / Gemini samples utilisent souvent video/mp4 pour les URLs vidéo
                        mime_type="video/mp4",
                    ),
                    prompt,
                ],
            )
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            logger.error(f"Gemini API failed (tentative {attempt + 1}/{GEMINI_SEGMENT_RETRIES}): {e}")
            # 400 INVALID_ARGUMENT arrive parfois de façon transitoire sur les URLs YouTube
            transient = (
                "INVALID_ARGUMENT" in msg or "invalid argument" in msg.lower()
                or "400" in msg or "RESOURCE_EXHAUSTED" in msg
            )
            if attempt + 1 < GEMINI_SEGMENT_RETRIES and transient:
                delay = 2.0 ** (attempt + 1)
                logger.warning(f"Réessai Gemini dans {delay:.0f}s…")
                time.sleep(delay)

    if response is None:
        if last_err and "INVALID_ARGUMENT" in str(last_err) and GEMINI_MODEL.lower().find("preview") >= 0:
            logger.warning(
                "Si l’erreur persiste avec un modèle *preview*, essaye GEMINI_MODEL=gemini-2.0-flash "
                "dans ton .env (meilleure prise en charge URL YouTube)."
            )
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
            # Gemini provides end — clamp duration to [CLIP_MIN_DURATION, CLIP_MAX_DURATION]
            end_raw = seg.get("end")
            if end_raw is not None:
                end = float(end_raw)
                duration = end - start
                if duration < CLIP_MIN_DURATION:
                    end = start + CLIP_MIN_DURATION
                elif duration > CLIP_MAX_DURATION:
                    end = start + CLIP_MAX_DURATION
            else:
                # Fallback: no end returned — use max duration
                end = start + CLIP_MAX_DURATION

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
