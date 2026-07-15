"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Models
    whisper_model: str = Field("large-v2", alias="WHISPER_MODEL")
    sentiment_model: str = Field(
        "cardiffnlp/twitter-roberta-base-sentiment-latest", alias="SENTIMENT_MODEL"
    )
    emotion_model: str = Field(
        "j-hartmann/emotion-english-distilroberta-base", alias="EMOTION_MODEL"
    )
    device: str = Field("auto", alias="DEVICE")
    compute_type: str = Field("float16", alias="COMPUTE_TYPE")

    # API
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    max_upload_mb: int = Field(50, alias="MAX_UPLOAD_MB")

    # Preprocessing
    target_sr: int = Field(16000, alias="TARGET_SR")
    vad_aggressiveness: int = Field(2, alias="VAD_AGGRESSIVENESS")

    @property
    def resolved_device(self) -> str:
        return _resolve_device(self.device)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
