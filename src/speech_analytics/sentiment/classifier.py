"""Transformer-based sentiment classification (RoBERTa / BERT).

Uses Hugging Face ``transformers`` text-classification pipeline. Falls back to a
lightweight lexicon scorer when transformers is unavailable, keeping the API
and pipeline runnable in minimal environments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import settings

log = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    _HAS_TRANSFORMERS = False

_LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

_POS = {"good", "great", "excellent", "love", "happy", "amazing", "wonderful", "best", "fantastic", "glad", "pleased"}
_NEG = {"bad", "terrible", "hate", "awful", "worst", "angry", "sad", "poor", "horrible", "disappointed", "frustrated"}


@dataclass
class SentimentResult:
    label: str
    score: float
    distribution: dict[str, float] = field(default_factory=dict)
    model: str = ""


class SentimentClassifier:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.sentiment_model
        self.device = device or settings.resolved_device
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        if not _HAS_TRANSFORMERS:
            log.warning("transformers not installed; using lexicon sentiment fallback.")
            self._pipe = "lexicon"
            return self._pipe
        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        device_id = 0 if self.device == "cuda" else -1
        self._pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device_id, top_k=None, truncation=True
        )
        return self._pipe

    def _lexicon(self, text: str) -> SentimentResult:
        tokens = [t.strip(".,!?;:").lower() for t in text.split()]
        pos = sum(t in _POS for t in tokens)
        neg = sum(t in _NEG for t in tokens)
        total = pos + neg
        if total == 0:
            dist = {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
        else:
            p, n = pos / total, neg / total
            neu = max(0.0, 1 - abs(p - n))
            s = p + n + neu
            dist = {"negative": n / s, "neutral": neu / s, "positive": p / s}
        label = max(dist, key=dist.get)
        return SentimentResult(label=label, score=round(dist[label], 4), distribution=dist, model="lexicon")

    def predict(self, text: str) -> SentimentResult:
        pipe = self.load()
        if not text or not text.strip():
            return SentimentResult(label="neutral", score=1.0, distribution={"neutral": 1.0}, model=self.model_name)
        if pipe == "lexicon":
            return self._lexicon(text)

        scores = pipe(text)[0]  # list of {label, score}
        dist = {_LABEL_MAP.get(d["label"].lower(), d["label"].lower()): float(d["score"]) for d in scores}
        top = max(dist, key=dist.get)
        return SentimentResult(
            label=top,
            score=round(dist[top], 4),
            distribution={k: round(v, 4) for k, v in dist.items()},
            model=self.model_name,
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.predict(t) for t in texts]
