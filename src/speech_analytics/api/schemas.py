"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentOut(BaseModel):
    label: str
    score: float
    model: str = ""


class EmotionOut(BaseModel):
    label: str
    score: float
    model: str = ""


class AudioMeta(BaseModel):
    duration_s: float
    sample_rate: int
    speech_ratio: float
    steps: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    start: float
    end: float
    text: str


class AnalyzeResponse(BaseModel):
    transcript: str
    language: str
    sentiment: SentimentOut
    emotion: EmotionOut
    audio: AudioMeta
    segments: list[Segment] = Field(default_factory=list)
    latency_ms: float


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    segments: list[Segment] = Field(default_factory=list)
    latency_ms: float


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: SentimentOut
    emotion: EmotionOut


class HealthResponse(BaseModel):
    status: str
    device: str
    whisper_model: str
    sentiment_model: str
