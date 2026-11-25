# 6. Agent runs and batch jobs use a thread pool, not a process pool

Date: 2026-11-25 · Status: accepted

## Context
The spec asks for a background job interface with a local process-pool implementation. An agent run pauses for
approvals in the middle of a stage (cleaning plan, final acceptance), streams fine-grained progress (every CV fold and
Optuna trial) and keeps large in-memory objects between stages (the prepared task, round-1 runs with out-of-fold
predictions) that are expensive to pickle across processes.

## Decision
- `jobs/runner.py` defines a `JobRunner` protocol; `ThreadJobRunner` (a `ThreadPoolExecutor`) is the local implementation.
- The heavy numerical work (NumPy, scikit-learn, LightGBM, XGBoost) releases the GIL, and model `n_jobs` is capped at
  `cpu_count - 1` (max 4), so the API stays responsive during training.
- Each run checkpoints its state to the `agent_sessions.state` column after every stage. A run interrupted by a restart
  becomes `failed / interrupted / retryable`, and `POST /sessions/{id}/retry` resumes from that stage (stages that need
  in-memory results from an earlier stage step back to it automatically).
- `agent/context.py` (the `StageContext` every stage uses) is added next to `orchestrator.py`.

## Consequences
A crash inside native code can take the API process down with it. A process- or queue-backed runner can replace
`ThreadJobRunner` later without changing the stages, because they only talk to `StageContext`.
