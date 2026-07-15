from speech_analytics.analytics import aggregate, save_matplotlib_summary
from speech_analytics.pipeline import SpeechAnalyticsPipeline


def test_pipeline_end_to_end(sample_wav):
    pipe = SpeechAnalyticsPipeline()
    result = pipe.analyze(sample_wav)
    d = result.to_dict()
    assert "transcript" in d
    assert "sentiment" in d and "emotion" in d
    assert d["audio"]["sample_rate"] == 16000


def test_aggregate(sample_wav):
    pipe = SpeechAnalyticsPipeline()
    results = [pipe.analyze(sample_wav) for _ in range(3)]
    agg = aggregate(results)
    assert agg["count"] == 3
    assert "sentiment_counts" in agg
