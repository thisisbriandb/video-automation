FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required by yt-dlp-ejs for YouTube JS challenge solving)
RUN curl -fsSL https://dl.deno.land/release/latest/deno-x86_64-unknown-linux-gnu.zip \
    -o /tmp/deno.zip \
    && apt-get update && apt-get install -y --no-install-recommends unzip \
    && unzip /tmp/deno.zip -d /usr/local/bin/ \
    && rm /tmp/deno.zip \
    && apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
    && chmod +x /usr/local/bin/deno \
    && deno --version

WORKDIR /app

# Install Python dependencies (yt-dlp[default] pulls in yt-dlp-ejs for YouTube JS challenges)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import yt_dlp; print('yt-dlp version:', yt_dlp.version.__version__)" \
    && python -c "import yt_dlp_ejs; print('yt-dlp-ejs OK')"

# Copy app code
COPY video_maker/ video_maker/

# Copy YouTube cookies if present
COPY cookies.txt /app/cookies.txt

# Create working directories
RUN mkdir -p workdir output

# Railway sets PORT env var
ENV PORT=8001
ENV NUM_WORKERS=4
ENV WHISPER_MODEL=small
ENV FFMPEG_DIR=/usr/bin
ENV YOUTUBE_COOKIES_FILE=/app/cookies.txt

EXPOSE 8001

CMD uvicorn video_maker.app:app --host 0.0.0.0 --port $PORT
