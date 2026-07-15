# CPU-friendly base; swap for an nvidia/cuda image for GPU inference.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PYTHONPATH=/app/src

# ffmpeg is required by Whisper/pydub for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "speech_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
