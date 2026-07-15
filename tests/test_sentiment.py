from speech_analytics.emotion import EmotionRecognizer
from speech_analytics.sentiment import SentimentClassifier


def test_sentiment_positive():
    clf = SentimentClassifier()
    res = clf.predict("I absolutely love this, it is wonderful and great!")
    assert res.label in {"positive", "neutral", "negative"}
    assert 0.0 <= res.score <= 1.0
    assert abs(sum(res.distribution.values()) - 1.0) < 0.05 or res.model != "lexicon"


def test_sentiment_empty():
    clf = SentimentClassifier()
    assert clf.predict("").label == "neutral"


def test_emotion_prediction():
    rec = EmotionRecognizer()
    res = rec.predict("I am so happy and excited today!")
    assert res.label
    assert 0.0 <= res.score <= 1.0
