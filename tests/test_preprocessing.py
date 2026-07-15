import numpy as np

from speech_analytics.preprocessing import AudioPreprocessor, load_audio


def test_load_audio(sample_wav):
    audio, sr = load_audio(sample_wav, target_sr=16000)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_normalize_peaks_below_one():
    pre = AudioPreprocessor()
    x = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = pre.normalize(x)
    assert np.max(np.abs(out)) <= 0.98


def test_process_returns_result(sample_wav):
    pre = AudioPreprocessor()
    res = pre.process(sample_wav)
    assert res.sample_rate == 16000
    assert res.duration_s > 0
    assert 0.0 <= res.speech_ratio <= 1.0
    assert "normalize" in res.steps
