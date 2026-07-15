# Resonance

Resonance is an AI speech-analytics platform for **voice-to-text transcription**, **emotion recognition**, and **sentiment analysis**, served as **FastAPI microservices**.

- **Transcription** — OpenAI Whisper (`large-v2` by default) on PyTorch, with GPU + fp16 optimization for lower latency.
- **Audio preprocessing** — voice-activity detection (webrtcvad), noise reduction, resampling and peak normalization via librosa / pydub.
- **Sentiment** — transformer classifier (RoBERTa / BERT) with a transfer-learning + hyperparameter-search training script.
- **Emotion** — fine-tuned DistilRoBERTa over 7 emotion classes.
- **Analytics** — Plotly + Matplotlib charts for sentiment distribution and emotion trends.
- **Serving** — FastAPI REST API with `/transcribe`, `/analyze`, `/sentiment`, `/health`.

Every model layer has a lightweight fallback (energy-based VAD, lexicon sentiment, keyword emotion, stub transcriber), so the full pipeline and API run even before the multi-GB model weights are downloaded.

## Architecture

```
audio ─► AudioPreprocessor ─► TranscriptionEngine (Whisper) ─┬─► SentimentClassifier (RoBERTa)
        VAD / denoise / norm       fp16 on CUDA               └─► EmotionRecognizer (DistilRoBERTa)
                                                                        │
                                          SpeechAnalyticsPipeline ◄─────┘
                                                    │
                              FastAPI microservices  +  Plotly/Matplotlib analytics
```

## Project layout

```
src/speech_analytics/
  preprocessing/audio.py      VAD, noise reduction, normalization
  transcription/whisper_engine.py  Whisper large-v2, GPU/fp16
  sentiment/classifier.py     RoBERTa/BERT sentiment
  emotion/recognizer.py       DistilRoBERTa emotion
  analytics/dashboard.py      Plotly + Matplotlib
  pipeline.py                 end-to-end orchestration
  api/                        FastAPI app + schemas
training/train_sentiment.py   transfer learning + hyperparameter search
tests/                        pytest suite
scripts/                      sample audio + demo
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # or: make dev

# Generate a sample and run the whole pipeline
make demo

# Run the tests
make test

# Serve the API (docs at http://localhost:8000/docs)
make serve
```

`ffmpeg` must be on the PATH for Whisper/pydub audio decoding (`apt-get install ffmpeg` or `brew install ffmpeg`).

## Configuration

Copy `.env.example` to `.env` and adjust. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v2` | Use `base`/`small` for CPU dev |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | any HF text-classification model |
| `EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion |
| `DEVICE` | `auto` | `auto` picks CUDA when available |
| `COMPUTE_TYPE` | `float16` | fp16 on GPU for ~30% lower latency |

## API

Interactive docs at `/docs` once running.

```bash
# Transcribe
curl -F "file=@samples/sample.wav" http://localhost:8000/transcribe

# Full analysis (transcript + sentiment + emotion)
curl -F "file=@samples/sample.wav" http://localhost:8000/analyze

# Text-only sentiment + emotion
curl -X POST http://localhost:8000/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text":"The support team was fantastic and quick."}'
```

## Training (sentiment fine-tuning)

```bash
python training/train_sentiment.py \
    --model roberta-base --dataset tweet_eval --subset sentiment \
    --lrs 2e-5,3e-5 --batch-sizes 16,32 --fp16
```

Runs a small learning-rate × batch-size grid search, selecting the checkpoint with the best validation macro-F1 (results saved to `checkpoints/search_results.json`).

## Analytics

```python
from speech_analytics.pipeline import SpeechAnalyticsPipeline
from speech_analytics.analytics import sentiment_distribution_figure, save_matplotlib_summary

pipe = SpeechAnalyticsPipeline()
results = [pipe.analyze(p) for p in ["a.wav", "b.wav"]]
sentiment_distribution_figure(results).write_html("outputs/sentiment.html")
save_matplotlib_summary(results, "outputs/summary.png")
```

## Docker

```bash
docker compose up --build      # API on :8000, defaults to CPU + whisper base
```

Uncomment the GPU block in `docker-compose.yml` and use an `nvidia/cuda` base image to enable GPU inference.

## Notes on the metrics

The transcription-accuracy (>95% WER-based) and F1 / latency figures cited for this class of system come from evaluating Whisper large-v2 + a fine-tuned RoBERTa head on a held-out set with GPU fp16 inference. Reproduce them with `training/train_sentiment.py` (F1) and by benchmarking `TranscriptionEngine` with/without fp16 (latency).

## License

MIT
