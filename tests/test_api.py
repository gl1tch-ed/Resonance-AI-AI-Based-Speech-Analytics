import io
import struct
import wave

from fastapi.testclient import TestClient

from speech_analytics.api.main import app

client = TestClient(app)


def _wav_bytes(seconds=1, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(seconds * sr):
            frames += struct.pack("<h", int(0.2 * ((i % 100) / 100 - 0.5) * 32767))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sentiment_endpoint():
    r = client.post("/sentiment", json={"text": "This is great, I love it!"})
    assert r.status_code == 200
    body = r.json()
    assert "sentiment" in body and "emotion" in body


def test_analyze_endpoint():
    files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
    r = client.post("/analyze", files=files)
    assert r.status_code == 200
    body = r.json()
    assert "transcript" in body
    assert body["audio"]["sample_rate"] == 16000


def test_empty_upload_rejected():
    files = {"file": ("empty.wav", b"", "audio/wav")}
    r = client.post("/transcribe", files=files)
    assert r.status_code == 400
