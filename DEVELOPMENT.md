# Development notes

Setup and workflow notes for working on Resonance locally.

## Environment

- **OS:** developed on Windows (PowerShell), but the code is cross-platform.
- **Python:** 3.10+ (`py -m venv`). Note: on 3.13+ the stdlib `platform`
  module probes the OS version via WMI, which can hang `import torch` on some
  Windows boxes — 3.10 avoids that path, so it's the recommended interpreter here.
- Source lives under `src/`, so `PYTHONPATH` must include `src` (already set in
  `pyproject.toml` for pytest; set it manually for scripts/uvicorn).

## First-time setup

From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Two external requirements pip does not cover:

1. **ffmpeg** on PATH (Whisper/pydub decode audio through it):
   `winget install Gyan.FFmpeg`, then restart the terminal.
2. **torch** — if the default install is slow or fails, use the official index:
   - CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - CUDA (NVIDIA GPU): pick the matching wheel from https://pytorch.org

### Windows gotcha: webrtcvad

`webrtcvad` needs a C++ build toolchain and fails to build on a clean Windows
box. This repo uses **`webrtcvad-wheels`** instead (drop-in, prebuilt wheels,
still imports as `webrtcvad`). If you ever see a "Failed building wheel for
webrtcvad" error, make sure `requirements.txt` references `webrtcvad-wheels`,
not `webrtcvad`. It is optional — the preprocessor falls back to an
energy-based VAD when it is absent.

## Common commands

Set `PYTHONPATH` first in each new terminal: `$env:PYTHONPATH="src"`

```powershell
python scripts/generate_sample_audio.py          # writes samples/sample.wav
python scripts/demo.py samples/sample.wav         # full pipeline on a sample
pytest                                            # test suite
uvicorn speech_analytics.api.main:app --reload --port 8000   # API + docs at /docs
```

A `Makefile` mirrors these (`make demo`, `make test`, `make serve`).

### Fast, offline-friendly runs

The first real run downloads model weights; `large-v2` is multi-GB. For a quick
smoke test on CPU, shrink the model first:

```powershell
$env:WHISPER_MODEL="base"
$env:DEVICE="cpu"
$env:COMPUTE_TYPE="float32"
```

## Audio format support

Decoding goes through `libsndfile` (WAV, MP3, FLAC, OGG, AIFF, …). Apple's
**m4a/AAC is not supported** — transcode first:

```powershell
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

## Conventions

- Models are loaded lazily and cached on the instance; the API shares one
  `SpeechAnalyticsPipeline` via a FastAPI dependency. Call `pipeline.warmup()`
  to preload.
- Config is environment-driven. Copy `.env.example` to `.env` to override
  models, device, ports, upload limits. Don't hard-code model names — read from
  `settings`.
- Keep the graceful-fallback pattern when adding a model-backed component: guard
  heavy imports in `try/except` and provide a lightweight path so imports never
  hard-fail.
- Tests must not require network or GPU. They run against the fallbacks.

## Verifying changes

```powershell
$env:PYTHONPATH="src"
pytest
ruff check src tests    # optional lint
```
