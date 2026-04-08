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

# Install Node.js 20 directly (needed by yt-dlp for YouTube JS challenge solving)
RUN curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz \
    -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
