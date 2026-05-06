"""Point d’entrée Uvicorn : `python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload`

Utiliser `main:app`, pas `main.py:app` (le `.py` n’est pas un nom de module Python).
"""

from video_maker.app import app

__all__ = ["app"]
