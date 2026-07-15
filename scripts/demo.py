"""Minimal end-to-end demo of the pipeline on a sample WAV."""
from __future__ import annotations

import json
import sys

from speech_analytics.pipeline import SpeechAnalyticsPipeline


def main(audio_path: str) -> None:
    pipe = SpeechAnalyticsPipeline()
    result = pipe.analyze(audio_path)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/sample.wav")
