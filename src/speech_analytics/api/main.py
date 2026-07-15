"""FastAPI microservice exposing transcription and sentiment/emotion insights.

Endpoints
---------
GET  /health              service + model status
POST /transcribe          audio file -> transcript
POST /analyze             audio file -> transcript + sentiment + emotion
POST /sentiment           raw text   -> sentiment + emotion

Run: uvicorn speech_analytics.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from ..config import settings
from ..pipeline import SpeechAnalyticsPipeline
from . import schemas

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speech_analytics.api")

_pipeline: SpeechAnalyticsPipeline | None = None


def get_pipeline() -> SpeechAnalyticsPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SpeechAnalyticsPipeline()
    return _pipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title="Speech Analytics Platform",
        version="0.1.0",
        description="Transcription, emotion recognition and sentiment analysis microservices.",
    )

    @app.get("/health", response_model=schemas.HealthResponse)
    def health():
        return schemas.HealthResponse(
            status="ok",
            device=settings.resolved_device,
            whisper_model=settings.whisper_model,
            sentiment_model=settings.sentiment_model,
        )

    async def _read_upload(file: UploadFile) -> bytes:
        data = await file.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
        if not data:
            raise HTTPException(400, "Empty upload")
        return data

    @app.post("/transcribe", response_model=schemas.TranscribeResponse)
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        pipe: SpeechAnalyticsPipeline = Depends(get_pipeline),
    ):
        data = await _read_upload(file)
        pre = pipe.preprocessor.process(data)
        tr = pipe.transcriber.transcribe(pre.audio, pre.sample_rate, language=language)
        return schemas.TranscribeResponse(
            transcript=tr.text, language=tr.language, segments=tr.segments, latency_ms=tr.latency_ms
        )

    @app.post("/analyze", response_model=schemas.AnalyzeResponse)
    async def analyze(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        pipe: SpeechAnalyticsPipeline = Depends(get_pipeline),
    ):
        data = await _read_upload(file)
        result = pipe.analyze(data, language=language)
        return schemas.AnalyzeResponse(**result.to_dict())

    @app.post("/sentiment", response_model=schemas.SentimentResponse)
    def sentiment(
        req: schemas.SentimentRequest,
        pipe: SpeechAnalyticsPipeline = Depends(get_pipeline),
    ):
        s = pipe.sentiment.predict(req.text)
        e = pipe.emotion.predict(req.text)
        return schemas.SentimentResponse(
            sentiment=schemas.SentimentOut(**s.__dict__),
            emotion=schemas.EmotionOut(**e.__dict__),
        )

    return app


app = create_app()
