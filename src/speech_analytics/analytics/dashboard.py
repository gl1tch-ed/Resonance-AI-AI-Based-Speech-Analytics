"""Analytics + visualization for a batch of AnalyticsResults.

Plotly for interactive charts (sentiment distribution, emotion trend), and a
Matplotlib static summary for reports. Both degrade gracefully if the optional
plotting libs are missing.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def _as_dicts(results: Iterable) -> list[dict]:
    out = []
    for r in results:
        out.append(r.to_dict() if hasattr(r, "to_dict") else dict(r))
    return out


def aggregate(results: Iterable) -> dict:
    """Summarize sentiment/emotion counts across many analyses."""
    rows = _as_dicts(results)
    sent = Counter(r["sentiment"]["label"] for r in rows)
    emo = Counter(r["emotion"]["label"] for r in rows)
    n = len(rows) or 1
    avg_latency = sum(r.get("latency_ms", 0) for r in rows) / n
    return {
        "count": len(rows),
        "sentiment_counts": dict(sent),
        "emotion_counts": dict(emo),
        "avg_latency_ms": round(avg_latency, 1),
    }


def sentiment_distribution_figure(results: Iterable):
    import plotly.graph_objects as go

    agg = aggregate(results)
    counts = agg["sentiment_counts"]
    order = ["negative", "neutral", "positive"]
    labels = [l for l in order if l in counts] + [l for l in counts if l not in order]
    values = [counts[l] for l in labels]
    colors = {"negative": "#e45756", "neutral": "#9aa0a6", "positive": "#54a24b"}
    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=[colors.get(l, "#4c78a8") for l in labels])
    )
    fig.update_layout(title="Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Utterances")
    return fig


def emotion_trend_figure(results: Iterable):
    """Line chart of emotion scores over the sequence of analyses (time order)."""
    import plotly.graph_objects as go

    rows = _as_dicts(results)
    emotions = sorted({r["emotion"]["label"] for r in rows})
    x = list(range(1, len(rows) + 1))
    fig = go.Figure()
    for emo in emotions:
        y = [r["emotion"]["distribution"].get(emo, 0.0) for r in rows]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=emo))
    fig.update_layout(title="Emotion Trend", xaxis_title="Utterance #", yaxis_title="Score")
    return fig


def save_matplotlib_summary(results: Iterable, path: str = "outputs/summary.png") -> str:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate(results)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s = agg["sentiment_counts"]
    axes[0].bar(list(s.keys()), list(s.values()), color="#4c78a8")
    axes[0].set_title("Sentiment Distribution")
    e = agg["emotion_counts"]
    axes[1].bar(list(e.keys()), list(e.values()), color="#f58518")
    axes[1].set_title("Emotion Distribution")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
