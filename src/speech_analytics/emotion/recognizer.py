"""Text-based emotion recognition using a fine-tuned DistilRoBERTa classifier.

Returns the dominant emotion plus a full distribution over the 7 Ekman-style
classes. Falls back to a keyword heuristic when transformers is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import settings

log = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_pipeline

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    _HAS_TRANSFORMERS = False

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_KEYWORDS = {
    "joy": {"happy", "glad", "love", "great", "excited", "delighted", "wonderful"},
    "anger": {"angry", "furious", "mad", "annoyed", "hate", "outraged"},
    "sadness": {"sad", "unhappy", "depressed", "cry", "disappointed", "sorry"},
    "fear": {"afraid", "scared", "worried", "anxious", "nervous", "terrified"},
    "surprise": {"surprised", "shocked", "wow", "unexpected", "amazed"},
    "disgust": {"disgusting", "gross", "awful", "nasty", "revolting"},
}


@dataclass
class EmotionResult:
    label: str
    score: float
    distribution: dict[str, float] = field(default_factory=dict)
    model: str = ""


class EmotionRecognizer:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.emotion_model
        self.device = device or settings.resolved_device
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        if not _HAS_TRANSFORMERS:
            log.warning("transformers not installed; using keyword emotion fallback.")
            self._pipe = "keyword"
            return self._pipe
        device_id = 0 if self.device == "cuda" else -1
        self._pipe = hf_pipeline(
            "text-classification", model=self.model_name, top_k=None, device=device_id, truncation=True
        )
        return self._pipe

    def _keyword(self, text: str) -> EmotionResult:
        tokens = {t.strip(".,!?;:").lower() for t in text.split()}
        scores = {e: 0.0 for e in EMOTIONS}
        for emo, words in _KEYWORDS.items():
            scores[emo] = float(len(tokens & words))
        if sum(scores.values()) == 0:
            scores["neutral"] = 1.0
        total = sum(scores.values())
        dist = {k: round(v / total, 4) for k, v in scores.items()}
        label = max(dist, key=dist.get)
        return EmotionResult(label=label, score=dist[label], distribution=dist, model="keyword")

    def predict(self, text: str) -> EmotionResult:
        pipe = self.load()
        if not text or not text.strip():
            return EmotionResult(label="neutral", score=1.0, distribution={"neutral": 1.0}, model=self.model_name)
        if pipe == "keyword":
            return self._keyword(text)
        scores = pipe(text)[0]
        dist = {d["label"].lower(): round(float(d["score"]), 4) for d in scores}
        label = max(dist, key=dist.get)
        return EmotionResult(label=label, score=dist[label], distribution=dist, model=self.model_name)
