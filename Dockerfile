FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    xz-utils \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required by yt-dlp-ejs + bgutil PO Token provider)
RUN curl -fsSL -o /tmp/deno.zip \
       https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin/ \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && deno --version

WORKDIR /app

# Install Python dependencies (yt-dlp[default] pulls in yt-dlp-ejs for YouTube JS challenges)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import yt_dlp; print('yt-dlp version:', yt_dlp.version.__version__)" \
    && python -c "import yt_dlp_ejs; print('yt-dlp-ejs OK')" \
    && pip show bgutil-ytdlp-pot-provider | head -2

# Set up bgutil PO Token provider (generates Proof of Origin tokens to bypass YouTube bot detection)
RUN git clone --single-branch --branch 1.3.1 --depth 1 \
        https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /app/bgutil-ytdlp-pot-provider \
    && cd /app/bgutil-ytdlp-pot-provider/server \
    && deno install --allow-scripts=npm:canvas --frozen \
    && echo 'bgutil POT server built OK'

# Copy app code
COPY video_maker/ video_maker/

# Create working directories
RUN mkdir -p workdir output

# Railway sets PORT env var
ENV PORT=8001
ENV WHISPER_MODEL=small
ENV FFMPEG_DIR=/usr/bin
# Cookies: set YOUTUBE_COOKIES env var on Railway (base64-encoded Netscape cookies)
# The downloader auto-decodes and writes to a temp file at runtime

EXPOSE 8001

# Start bgutil POT server in background, then launch uvicorn
CMD cd /app/bgutil-ytdlp-pot-provider/server/node_modules && \
    deno run --allow-env --allow-net --allow-ffi=. --allow-read=. ../src/main.ts --port 4416 & \
    sleep 2 && \
    uvicorn video_maker.app:app --host 0.0.0.0 --port $PORT
