"""AI-driven speech analytics platform.

Voice-to-text transcription (Whisper large-v2), emotion recognition and
sentiment analysis (BERT / RoBERTa), served through FastAPI microservices.
"""
from .config import settings

__version__ = "0.1.0"
__all__ = ["settings", "__version__"]
