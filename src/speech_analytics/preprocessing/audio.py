"""Audio preprocessing: loading, resampling, noise filtering, normalization and VAD.

Built on librosa + pydub. Voice-activity detection uses webrtcvad when available
and falls back to an energy-based gate so the pipeline always runs.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa

    _HAS_LIBROSA = True
except Exception:  # pragma: no cover
    _HAS_LIBROSA = False

try:
    import noisereduce as nr

    _HAS_NR = True
except Exception:  # pragma: no cover
    _HAS_NR = False

try:
    import webrtcvad

    _HAS_WEBRTCVAD = True
except Exception:  # pragma: no cover
    _HAS_WEBRTCVAD = False


@dataclass
class PreprocessResult:
    """Container for a cleaned audio signal and its metadata."""

    audio: np.ndarray            # float32, mono, in [-1, 1]
    sample_rate: int
    duration_s: float
    speech_ratio: float          # fraction of frames flagged as speech
    steps: list[str] = field(default_factory=list)


def load_audio(path_or_bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file (path or raw bytes) as mono float32 at ``target_sr``."""
    if _HAS_LIBROSA:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            data, sr = librosa.load(io.BytesIO(path_or_bytes), sr=target_sr, mono=True)
        else:
            data, sr = librosa.load(path_or_bytes, sr=target_sr, mono=True)
        return data.astype(np.float32), sr
    # Fallback: stdlib wave reader (16-bit PCM WAV only)
    raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            pcm = pcm.reshape(-1, wf.getnchannels()).mean(axis=1)
    if sr != target_sr:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * target_sr / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
        sr = target_sr
    return pcm.astype(np.float32), sr


class AudioPreprocessor:
    """Configurable preprocessing chain producing clean 16 kHz mono audio."""

    def __init__(
        self,
        target_sr: int = 16000,
        vad_aggressiveness: int = 2,
        frame_ms: int = 30,
    ) -> None:
        self.target_sr = target_sr
        self.vad_aggressiveness = vad_aggressiveness
        self.frame_ms = frame_ms

    # -- individual steps -------------------------------------------------
    def denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if _HAS_NR:
            return nr.reduce_noise(y=audio, sr=sr).astype(np.float32)
        # Fallback: simple spectral high-pass to attenuate low-freq hum
        if _HAS_LIBROSA:
            return librosa.effects.preemphasis(audio).astype(np.float32)
        return audio

    @staticmethod
    def normalize(audio: np.ndarray, peak: float = 0.97) -> np.ndarray:
        m = float(np.max(np.abs(audio))) or 1.0
        return (audio / m * peak).astype(np.float32)

    def voice_activity(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """Return speech-only audio and the speech ratio."""
        frame_len = int(sr * self.frame_ms / 1000)
        if frame_len <= 0 or len(audio) < frame_len:
            return audio, 1.0

        if _HAS_WEBRTCVAD and sr in (8000, 16000, 32000, 48000):
            vad = webrtcvad.Vad(self.vad_aggressiveness)
            pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            flags, kept = [], []
            for start in range(0, len(pcm16) - frame_len, frame_len):
                frame = pcm16[start : start + frame_len]
                is_speech = vad.is_speech(frame.tobytes(), sr)
                flags.append(is_speech)
                if is_speech:
                    kept.append(audio[start : start + frame_len])
            ratio = float(np.mean(flags)) if flags else 1.0
            speech = np.concatenate(kept) if kept else audio
            return speech.astype(np.float32), ratio

        # Energy-gate fallback
        n_frames = len(audio) // frame_len
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.sqrt((frames**2).mean(axis=1) + 1e-9)
        thresh = max(0.5 * energy.mean(), 1e-3)
        mask = energy > thresh
        ratio = float(mask.mean()) if n_frames else 1.0
        speech = frames[mask].reshape(-1) if mask.any() else audio
        return speech.astype(np.float32), ratio

    # -- orchestration ----------------------------------------------------
    def process(self, path_or_bytes, apply_vad: bool = True) -> PreprocessResult:
        steps: list[str] = []
        audio, sr = load_audio(path_or_bytes, self.target_sr)
        steps.append(f"load@{sr}Hz")

        audio = self.denoise(audio, sr)
        steps.append("denoise")

        speech_ratio = 1.0
        if apply_vad:
            audio, speech_ratio = self.voice_activity(audio, sr)
            steps.append("vad")

        audio = self.normalize(audio)
        steps.append("normalize")

        return PreprocessResult(
            audio=audio,
            sample_rate=sr,
            duration_s=round(len(audio) / sr, 3),
            speech_ratio=round(speech_ratio, 3),
            steps=steps,
        )
