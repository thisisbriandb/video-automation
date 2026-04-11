# Video Maker – YouTube → Shorts (TikTok / Reels)

Transforme automatiquement des vidéos YouTube longues (16:9) en clips courts viraux (9:16) avec sous-titres style TikTok, recadrage automatique sur le visage et normalisation audio.

**Aucune API payante requise** — tout tourne en local avec Whisper (IA de transcription) et OpenCV (détection visuelle).

---

## Fonctionnalités

- **Détection automatique des meilleurs moments** — analyse audio (énergie, rythme) + visuelle (mouvement, visages)
- **Recadrage dynamique 9:16** — le cadre suit le visage du speaker
- **Sous-titres style TikTok** — gros, centrés, incrustés dans la vidéo
- **Normalisation audio** — volume constant entre les clips
- **Interface web** — colle un lien YouTube, attends, télécharge tes clips

---

# 🚀 Guide d'installation (Windows)

> Ce guide est fait pour quelqu'un qui n'a jamais touché à du code.
> Suis chaque étape dans l'ordre. En cas de doute, redémarre ton PC et recommence l'étape qui bloque.

---

## Étape 1 : Installer Python 3.11

1. Va sur **https://www.python.org/downloads/**
2. Clique sur **"Download Python 3.11.x"** (ou la version 3.11 la plus récente)
3. Lance l'installateur `.exe`
4. **IMPORTANT** : Coche la case **"Add Python to PATH"** en bas de la fenêtre
5. Clique sur **"Install Now"**
6. Attends la fin, puis ferme l'installateur

**Vérification** — Ouvre un terminal (appuie `Windows + R`, tape `cmd`, Entrée) :
```
python --version
```
Tu dois voir `Python 3.11.x`. Si tu vois une erreur, redémarre ton PC et réessaie.

---

## Étape 2 : Installer FFmpeg

FFmpeg est le logiciel qui découpe et assemble les vidéos.

1. Ouvre un terminal (`Windows + R` → `cmd` → Entrée)
2. Tape cette commande :
```
winget install Gyan.FFmpeg
```
3. Appuie sur Entrée et attends la fin
4. **Ferme le terminal et rouvre-en un nouveau**

**Vérification** :
```
ffmpeg -version
```
Tu dois voir un numéro de version. Si ça ne marche pas, redémarre ton PC.

> **Alternative** : Si `winget` ne marche pas, télécharge FFmpeg manuellement sur https://www.gyan.dev/ffmpeg/builds/ (prends "ffmpeg-release-full.7z"), décompresse-le, et note le chemin du dossier `bin` — tu en auras besoin dans l'Étape 6.

---

## Étape 3 : Installer Deno

Deno est un moteur JavaScript nécessaire pour que le téléchargement YouTube fonctionne.

1. Ouvre un terminal (`Windows + R` → `cmd` → Entrée)
2. Tape :
```
winget install DenoLand.Deno
```
3. Attends la fin, puis **ferme et rouvre le terminal**

**Vérification** :
```
deno --version
```
Tu dois voir un numéro de version.

---

## Étape 4 : Installer Git (si pas déjà fait)

Git permet de télécharger le code du projet.

1. Va sur **https://git-scm.com/download/win**
2. Télécharge et installe (laisse toutes les options par défaut)
3. Redémarre ton terminal

**Vérification** :
```
git --version
```

---

## Étape 5 : Télécharger le projet

1. Ouvre un terminal
2. Va dans le dossier où tu veux mettre le projet (par exemple le Bureau) :
```
cd %USERPROFILE%\Desktop
```
3. Clone le projet :
```
git clone https://github.com/thisisbriandb/video-automation.git video-maker
```
4. Entre dans le dossier :
```
cd video-maker
```

---

## Étape 6 : Installer les dépendances Python

Toujours dans le dossier `video-maker` :

```
python -m venv .venv
```
```
.venv\Scripts\activate
```

Ton terminal doit maintenant afficher `(.venv)` au début de la ligne. Ensuite :

```
pip install -r requirements.txt
```

Ça va prendre **plusieurs minutes** (téléchargement de PyTorch, Whisper, etc.). Patiente.

> Si tu vois des erreurs en rouge sur `torch` ou `torchaudio`, essaie :
> ```
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## Étape 7 : Configurer l'environnement

1. Dans le dossier `video-maker`, copie le fichier d'exemple :
```
copy .env.example .env
```

2. Ouvre le fichier `.env` avec le Bloc-notes :
```
notepad .env
```

3. Vérifie que le contenu ressemble à ça (tu peux ajuster les valeurs) :
```
YOUTUBE_COOKIES_FILE=./cookies.txt
WORKING_DIR=./workdir
OUTPUT_DIR=./output
MAX_CLIPS=5
CLIP_MIN_DURATION=60
CLIP_MAX_DURATION=90
NUM_WORKERS=4
WHISPER_MODEL=small
```

> **`NUM_WORKERS`** : mets `4` si ton PC a 8+ Go de RAM, sinon laisse `2`.
> **`WHISPER_MODEL`** : `tiny` (rapide, léger) / `base` (bon compromis) / `small` (meilleure qualité, lent). Mets `small` si tu as 8+ Go de RAM.

4. Sauvegarde (`Ctrl+S`) et ferme le Bloc-notes.

---

## Étape 8 : Extraire les cookies YouTube (Firefox)

Les cookies permettent de télécharger les vidéos YouTube sans être bloqué.

### Prérequis : Firefox avec un compte YouTube connecté

1. Ouvre **Firefox** (pas Chrome, pas Edge)
2. Va sur **https://www.youtube.com**
3. Connecte-toi avec ton compte Google
4. Regarde au moins une vidéo (pour activer la session)
5. **Ferme Firefox complètement** (important : le script a besoin d'accéder à la base de données)

### Extraction automatique

Retourne dans ton terminal (dans le dossier `video-maker`, avec `(.venv)` actif) :

```
python scripts/extract_cookies.py
```

Tu devrais voir quelque chose comme :
```
Found Firefox cookies DB: C:\Users\...\cookies.sqlite
Found 85 YouTube/Google cookies
Written 85 unique cookies to cookies.txt
```

Si ça marche, un fichier `cookies.txt` a été créé dans le dossier du projet.

### Alternative manuelle (si le script ne marche pas)

1. Installe l'extension Firefox **"cookies.txt"** : https://addons.mozilla.org/fr/firefox/addon/cookies-txt/
2. Va sur **youtube.com** (connecté)
3. Clique sur l'icône de l'extension → **"Export"** → **"Current site"**
4. Sauvegarde le fichier sous le nom `cookies.txt` dans le dossier `video-maker`

> **Les cookies expirent au bout de ~2 semaines.** Quand les vidéos ne se téléchargent plus, refais cette étape.

---

## Étape 9 : Lancer le serveur

Toujours dans le dossier `video-maker`, avec `(.venv)` actif :

```
python -m uvicorn video_maker.app:app --host 0.0.0.0 --port 8001 --reload
```

Tu devrais voir :
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Started reloader process
```

**Le serveur tourne !** Ne ferme pas ce terminal.

---

## Étape 10 : Utiliser l'application

1. Ouvre ton navigateur (Chrome, Firefox, peu importe)
2. Va sur **http://localhost:8001**
3. Colle un lien YouTube dans le champ
4. Clique sur **"Lancer"**
5. Attends — la progression s'affiche en temps réel :
   - Téléchargement de la vidéo
   - Analyse audio + visuelle
   - Transcription (sous-titres)
   - Rendu des clips
6. Quand c'est fini, **télécharge tes clips** directement depuis la page

> **Temps de traitement** : pour une vidéo de 20 min, compte ~10-30 min selon ton PC.
> La première utilisation est plus lente (téléchargement du modèle Whisper).

---

# 📋 Utilisation quotidienne

Une fois installé, voici ce que tu fais chaque jour :

### Option rapide : double-clic

1. Va dans le dossier `video-maker`
2. Double-clique sur **`start.bat`**
3. Le serveur démarre automatiquement
4. Ouvre ton navigateur → **http://localhost:8001**

### Option manuelle (terminal)

1. **Ouvre un terminal** (`Windows + R` → `cmd`)
2. **Va dans le dossier** :
   ```
   cd %USERPROFILE%\Desktop\video-maker
   ```
3. **Active l'environnement** :
   ```
   .venv\Scripts\activate
   ```
4. **Lance le serveur** :
   ```
   python -m uvicorn video_maker.app:app --host 0.0.0.0 --port 8001 --reload
   ```
5. **Ouvre le navigateur** → `http://localhost:8001`
6. Colle ton lien YouTube, lance, attends, télécharge

Pour arrêter le serveur : appuie `Ctrl+C` dans le terminal (ou ferme la fenêtre `start.bat`).

---

# ❓ Dépannage

### "python n'est pas reconnu"
→ Réinstalle Python en cochant **"Add to PATH"**, puis redémarre ton PC.

### "ffmpeg n'est pas reconnu"
→ Réinstalle FFmpeg avec `winget install Gyan.FFmpeg` et redémarre le terminal.
→ Si tu l'as installé manuellement, ajoute le chemin du dossier `bin` dans le fichier `.env` :
```
FFMPEG_DIR=C:\chemin\vers\ffmpeg\bin
```

### "pip install échoue sur torch"
→ Installe PyTorch manuellement d'abord :
```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Puis relance `pip install -r requirements.txt`.

### "Les vidéos ne se téléchargent pas" / "Sign in to confirm you're not a bot"
→ Tes cookies YouTube ont expiré. Refais l'**Étape 8**.

### "Le traitement est très lent"
→ Change `WHISPER_MODEL=tiny` dans le fichier `.env` (moins précis mais 5x plus rapide).
→ Réduis `MAX_CLIPS=3` pour générer moins de clips.

### "Erreur mémoire / le programme plante"
→ Réduis dans `.env` :
```
NUM_WORKERS=2
WHISPER_MODEL=tiny
MAX_CLIPS=3
```

### "Le terminal affiche plein d'erreurs rouges"
→ Copie le message d'erreur et envoie-le moi, je regarderai.

---

# ⚙️ Configuration complète (.env)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `YOUTUBE_COOKIES_FILE` | `./cookies.txt` | Chemin vers le fichier de cookies Firefox |
| `WORKING_DIR` | `./workdir` | Dossier temporaire (téléchargements, fichiers intermédiaires) |
| `OUTPUT_DIR` | `./output` | Dossier des clips finaux rendus |
| `FFMPEG_DIR` | *(auto-détecté)* | Chemin vers le dossier contenant `ffmpeg.exe` |
| `MAX_CLIPS` | `10` | Nombre max de clips à extraire par vidéo |
| `CLIP_MIN_DURATION` | `60` | Durée minimum d'un clip (secondes) |
| `CLIP_MAX_DURATION` | `90` | Durée maximum d'un clip (secondes) |
| `NUM_WORKERS` | `2` | Nombre de workers parallèles (scoring + extraction audio) |
| `RENDER_WORKERS` | `1` | Nombre de clips rendus en parallèle |
| `WHISPER_MODEL` | `tiny` | Modèle de transcription : `tiny` / `base` / `small` / `medium` |

---

# Architecture du projet

```
video-maker/
├── video_maker/              # Code principal
│   ├── app.py                # Serveur web (FastAPI)
│   ├── config.py             # Configuration (.env)
│   ├── downloader.py         # Téléchargement YouTube (yt-dlp)
│   ├── scorer.py             # Analyse audio (librosa) + visuelle (OpenCV)
│   ├── analyzer.py           # Orchestration scoring → Whisper → clips
│   ├── transcriber.py        # Transcription sous-titres (Whisper)
│   ├── renderer.py           # Rendu FFmpeg (crop, scale, sous-titres)
│   ├── vision.py             # Détection de visages (OpenCV)
│   ├── pipeline.py           # Pipeline principal (download → analyse → rendu)
│   ├── models.py             # Modèles de données
│   ├── utils.py              # Fonctions utilitaires
│   └── static/index.html     # Interface web
├── scripts/
│   └── extract_cookies.py    # Extracteur de cookies Firefox
├── start.bat                 # Double-clic pour lancer (Windows)
├── .env                      # Ta configuration locale
├── .env.example              # Exemple de configuration
├── cookies.txt               # Cookies YouTube (généré par le script)
├── requirements.txt          # Dépendances Python
├── Dockerfile                # Pour déploiement cloud (Railway)
└── README.md                 # Ce fichier
```

## Pipeline

1. **Téléchargement** — yt-dlp récupère la vidéo en 1080p
2. **Scoring** — Analyse audio (énergie, rythme, flux spectral) + visuelle (mouvement, détection de visages) sur des fenêtres de 30s
3. **Sélection** — Les meilleurs segments sont fusionnés et étendus à ≥60s
4. **Transcription** — Whisper génère les sous-titres mot par mot
5. **Rendu** — FFmpeg crop 9:16 centré sur le visage → scale 1080×1920 → sous-titres → normalisation audio → export MP4
6. **Nettoyage** — Suppression des fichiers temporaires
