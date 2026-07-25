# Port allocation — 33-sentiment-roberta

Authoritative scheme: [`../../PORT-ALLOCATION.md`](../../PORT-ALLOCATION.md). For project `NN=33`:
frontend `3NN0`, API `8NN0`, Postgres `54NN`, extras `9NN0`.

These are **placeholder/convention values**. Real deployment values, if any ever exist, belong in a
gitignored `ops/ports.local.md` and are never published.

## Assigned

| Service | Port | Bound by this repo? |
|---|---|---|
| Frontend / web UI | `3330` | **No** — reserved only |
| Backend API | `8330` | **No** — reserved only |
| PostgreSQL | `5433` | **No** — reserved only |
| Redis | — | n/a |
| MLflow UI (optional) | `9330` | Only if the optional tracking UI is used |

## Why most of these are unused

This is a Template A ML-research repo: a config-driven training entrypoint that writes to
`runs/run_N/` and a markdown report. There is no service to run. The ports above are **reserved so
that nothing in this repo can ever collide with another project on this machine**, not because a
server exists. Reserving a port and binding nothing is the correct outcome here — do not invent a
dashboard to justify a number in this table.

Experiment tracking is filesystem-first: `runs/run_N/{run_meta.json, metrics.json, predictions.parquet, figures/}`
with a `runs/latest` symlink. Zero dependencies, works offline, diffable in review.

## If the optional MLflow UI is used

```bash
MLFLOW_PORT=${MLFLOW_PORT:-9330}
mlflow ui --host 127.0.0.1 --port "$MLFLOW_PORT"
```

Bind `127.0.0.1` explicitly, never `0.0.0.0`.

## Rules

1. Never bind a framework default: `5432`, `6379`, `3000`, `8000`, `5000`, `7000`.
   `5000` and `7000` are taken by macOS AirPlay Receiver; `11434` by a local Ollama.
2. Express every port as an overridable env var: `MLFLOW_PORT=${MLFLOW_PORT:-9330}`.
3. Verify before declaring done — this must print nothing, or only this project:
   ```bash
   lsof -nP -iTCP:9330 -sTCP:LISTEN
   ```
