FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY video_maker/ video_maker/

# Create working directories
RUN mkdir -p workdir output

# Railway sets PORT env var
ENV PORT=8001
ENV NUM_WORKERS=4
ENV WHISPER_MODEL=small
ENV FFMPEG_DIR=/usr/bin

EXPOSE 8001

CMD uvicorn video_maker.app:app --host 0.0.0.0 --port $PORT
