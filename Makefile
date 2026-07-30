# Sentiment-RoBERTa: one command per pipeline stage.
# Every target below is exercised by scripts/verify_fresh_clone.sh or by CI.

.DEFAULT_GOAL := help
.PHONY: help setup data sample smoke dev small train ablation evidence model-evidence figures diagrams \
        diagrams-check report test quality-evidence lint format notebook verify clean all

UV ?= uv
PY := $(UV) run python
PYTHONHASHSEED ?= 1337
export PYTHONHASHSEED

# ── Reserved ports (docs/ports.example.md). Nothing in this repo binds any of them. ──
MLFLOW_PORT ?= 9330
PUBLISHED_RUN ?= runs/run_2
ABLATION_RUN ?= runs/run_3
SCHEDULE_RUN ?= runs/run_5
MODEL_EVIDENCE ?= reports/evidence/model_figures.json

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install pinned deps + pre-commit hooks
	$(UV) sync
	@if [ -d .git ]; then $(UV) run pre-commit install; fi

data:  ## Download the ungated amazon_polarity subset used by cfg/default.yaml
	$(PY) scripts/download_data.py --split train --rows 200000
	$(PY) scripts/download_data.py --split test  --rows 20000

sample:  ## Regenerate the committed synthetic smoke fixtures
	$(PY) scripts/make_sample.py --n 1000 --n-test 400 --seed 1337

smoke:  ## End-to-end run: committed synthetic fixtures, random-weight model
	$(PY) train.py -c cfg/smoke.yaml

dev:  ## Quickstart run: 2,000 train / 500 test, 1 epoch, seq 128
	$(PY) train.py -c cfg/dev.yaml

small:  ## The published run: 9,000 / 1,000, seq 256, bounded by WALL_CLOCK_CAP_MIN
	$(PY) train.py -c cfg/small.yaml

train: small  ## Alias for `make small` (the config whose numbers are published)

ablation:  ## The 4-cell preprocessing ablation (stopwords x n-gram range)
	$(PY) train.py -c cfg/small.yaml -p cfg/baseline_ablation.json --baselines-only

evidence:  ## Export tracked, review-text-free evidence for the published runs
	$(PY) scripts/export_evidence.py $(PUBLISHED_RUN) $(ABLATION_RUN) \
	  run_5=$(SCHEDULE_RUN) -o reports/evidence

model-evidence:  ## Measure live weights into compact, review-text-free figure evidence
	$(PY) scripts/export_figures.py -i $(PUBLISHED_RUN) -a $(ABLATION_RUN) \
	  --write-model-evidence $(MODEL_EVIDENCE) -o docs/images --publish

figures:  ## Regenerate and publish all eleven PNGs from committed evidence
	$(PY) scripts/export_figures.py -i reports/evidence/run_2 -a reports/evidence/run_3 \
	  --model-evidence $(MODEL_EVIDENCE) -o docs/images --publish

diagrams:  ## Regenerate the committed SVG diagrams from their Markdown sources
	./scripts/export_diagrams.sh

diagrams-check:  ## Render every Mermaid source and verify every label fits its box
	$(PY) scripts/check_diagram_text.py

report:  ## Regenerate reports/RESULTS.md from the explicit published runs
	$(PY) evaluate.py -i $(PUBLISHED_RUN) -a $(ABLATION_RUN) -s $(SCHEDULE_RUN) \
	  -o reports/RESULTS.md

notebook:  ## Execute the narrative notebook and SAVE its outputs
	$(PY) scripts/run_notebook.py

test:  ## pytest with coverage on the pure-logic core
	$(UV) run pytest --cov --cov-report=term-missing

quality-evidence:  ## Measure and write README test-count and coverage evidence
	$(UV) run pytest -q --cov --cov-report=json:.pytest_cache/coverage.json \
	  --junitxml=.pytest_cache/junit.xml
	$(PY) scripts/check_quality_claims.py --write \
	  --coverage-json .pytest_cache/coverage.json --junit-xml .pytest_cache/junit.xml

lint:  ## ruff check + ruff format --check + mypy
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy .

format:  ## Apply ruff formatting and autofixes
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

verify:  ## Clone committed HEAD into a temp dir and run the documented quickstart
	./scripts/verify_fresh_clone.sh

clean:  ## Remove caches (never touches runs/ or data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint test smoke figures report  ## Lint, test, smoke-train, figures, report
