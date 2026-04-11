@echo off
echo ============================================
echo   Video Maker - Demarrage du serveur
echo ============================================
echo.

cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    echo ERREUR : L'environnement Python n'est pas installe.
    echo Suis le guide d'installation dans README.md (Etape 6).
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

if not exist .env (
    echo ERREUR : Le fichier .env n'existe pas.
    echo Copie .env.example en .env et configure-le (voir README.md Etape 7).
    pause
    exit /b 1
)

echo Serveur en cours de demarrage...
echo.
echo   Ouvre ton navigateur sur : http://localhost:8001
echo.
echo   Pour arreter : ferme cette fenetre ou appuie Ctrl+C
echo ============================================
echo.

python -m uvicorn video_maker.app:app --host 0.0.0.0 --port 8001 --reload

pause
