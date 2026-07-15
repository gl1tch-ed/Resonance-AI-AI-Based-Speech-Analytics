.PHONY: install dev sample demo test lint serve docker
install:
	pip install -r requirements.txt
dev:
	pip install -r requirements-dev.txt
sample:
	python scripts/generate_sample_audio.py
demo: sample
	PYTHONPATH=src python scripts/demo.py samples/sample.wav
test:
	PYTHONPATH=src pytest
lint:
	ruff check src tests
serve:
	PYTHONPATH=src uvicorn speech_analytics.api.main:app --reload --host 0.0.0.0 --port 8000
docker:
	docker compose up --build
