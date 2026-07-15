import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    sr = 16000
    path = tmp_path_factory.mktemp("audio") / "sample.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(sr):  # 1 second
            val = 0.3 * ((i % 200) / 200 - 0.5)
            frames += struct.pack("<h", int(val * 32767))
        wf.writeframes(bytes(frames))
    return str(path)
