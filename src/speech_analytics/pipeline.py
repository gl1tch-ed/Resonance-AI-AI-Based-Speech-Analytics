"""End-to-end speech analytics pipeline.

audio -> preprocess (VAD/denoise/normalize) -> transcribe (Whisper)
      -> sentiment (RoBERTa) + emotion (DistilRoBERTa) -> combined result.

Models are loaded lazily and shared across calls so the pipeline can back an
API process cheaply.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import settings
from .emotion import EmotionRecognizer
from .preprocessing import AudioPreprocessor
from .sentiment import SentimentClassifier
from .transcription import TranscriptionEngine


@dataclass
class AnalyticsResult:
    transcript: str
    language: str
    sentiment: dict
    emotion: dict
    audio: dict
    segments: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SpeechAnalyticsPipeline:
    def __init__(
        self,
        preprocessor: AudioPreprocessor | None = None,
        transcriber: TranscriptionEngine | None = None,
        sentiment: SentimentClassifier | None = None,
        emotion: EmotionRecognizer | None = None,
    ) -> None:
        self.preprocessor = preprocessor or AudioPreprocessor(
            target_sr=settings.target_sr, vad_aggressiveness=settings.vad_aggressiveness
        )
        self.transcriber = transcriber or TranscriptionEngine()
        self.sentiment = sentiment or SentimentClassifier()
        self.emotion = emotion or EmotionRecognizer()

    def warmup(self) -> None:
        """Preload all models (useful at API startup)."""
        self.transcriber.load()
        self.sentiment.load()
        self.emotion.load()

    def analyze(self, path_or_bytes, language: str | None = None, apply_vad: bool = True) -> AnalyticsResult:
        pre = self.preprocessor.process(path_or_bytes, apply_vad=apply_vad)
        tr = self.transcriber.transcribe(pre.audio, pre.sample_rate, language=language)
        sent = self.sentiment.predict(tr.text)
        emo = self.emotion.predict(tr.text)

        return AnalyticsResult(
            transcript=tr.text,
            language=tr.language,
            sentiment=asdict(sent),
            emotion=asdict(emo),
            audio={
                "duration_s": pre.duration_s,
                "sample_rate": pre.sample_rate,
                "speech_ratio": pre.speech_ratio,
                "steps": pre.steps,
            },
            segments=tr.segments,
            latency_ms=round(tr.latency_ms, 1),
        )
