"""Generate a short synthetic WAV sample (no external deps) for smoke tests."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def write_tone(path: str, seconds: float = 2.0, sr: int = 16000) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            # amplitude-modulated tone to give the VAD something to chew on
            env = 0.5 * (1 + math.sin(2 * math.pi * 2 * i / sr))
            val = env * 0.3 * math.sin(2 * math.pi * 220 * i / sr)
            frames += struct.pack("<h", int(val * 32767))
        wf.writeframes(bytes(frames))
    return path


if __name__ == "__main__":
    out = write_tone("samples/sample.wav")
    print(f"wrote {out}")
