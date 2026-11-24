# 5. Extra modules in the ML engine

Date: 2026-11-25 · Status: accepted

## Context
The spec lists the `ml/` modules (ingest, profiling, detection, strategy, preprocessing, features, candidates, training, critic, evaluation, explain, registry, inference). Building them surfaced pieces that do not belong in any single one of those.

## Decision
Four modules are added:

| Module | Why |
|---|---|
| `ml/types.py` | Shared dataclasses (`Detection`, `Strategy`, `TrainConfig`, `RunResult`, `Finding`, `CleaningStep`) used by every stage and by the agent checkpoint. Avoids circular imports. |
| `ml/metrics.py` | One place for metric computation, labels and display formatting, so the leaderboard, critic, final card and model card never disagree. |
| `ml/forecasting.py` | Forecasting is a different engine (panel preparation, lag features ≥ horizon, rolling-origin backtest, seasonal-naive baseline, residual-based intervals). Keeping it separate keeps `training.py` focused on tabular CV. |
| `ml/pipeline.py` | Stage functions (round 1, critic, round 2, final, artifact) that the agent calls one by one, plus `run_all()`, a straight-through runner used by the CLI (`python -m datalyon.cli`) and the sample integration tests. |

MLflow logging lives in `ml/registry.py` (`Tracker`), as the spec places it.

## Consequences
The agent orchestrator never re-implements ML logic; it sequences `pipeline.py` stages, adds approvals and narrates with the LLM.
