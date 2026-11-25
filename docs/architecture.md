# Architecture

Datalyon is a two-process app: a React single-page app and a FastAPI service. They talk over a versioned REST API (`/api/v1`) plus one Server-Sent Events stream per agent session. Everything runs locally: SQLite for metadata, the file system for uploads and model artifacts, and optional MLflow tracking.

```
┌──────────────────────────── apps/web (Vite · React 19 · TS) ───────────────────────────┐
│  Dashboard · Agent chat + workspace · Model catalog · Model detail (+ Playground)       │
│  Datasets                                                                               │
│                                                                                         │
│  TanStack Query ──► lib/apiClient (typed from openapi.json)                             │
│  EventSource   ──► lib/sse ──► features/agent/events (Zod) ──► agentReducer (pure)       │
└───────────────┬───────────────────────────────────────────────┬─────────────────────────┘
                │ REST /api/v1/*                                │ SSE /sessions/{id}/events
┌───────────────▼───────────────────────────────────────────────▼─────────────────────────┐
│ services/api (FastAPI · Pydantic v2)                                                    │
│                                                                                         │
│  api/routes ── datasets · sessions · models · predictions · activity · health          │
│      │                                                                                  │
│      ├── agent/orchestrator ── AgentService ── AgentRun (one thread per session)        │
│      │        │   StageContext: say · tool · card · ask · checkpoint · cancel           │
│      │        │   stages/: data → detect → strategy → prep → train → critic → final     │
│      │        │            → publish                                                     │
│      │        ├── agent/llm: GeminiLLM (JSON schema) ─fallback─► MockLLM (rules)        │
│      │        └── EventBus: persist to session_events + in-memory tail                  │
│      │                                                                                  │
│      ├── ml/ (pure library, no web or DB imports)                                       │
│      │     ingest · profiling · detection · strategy · features · preprocessing ·       │
│      │     candidates · training · critic · evaluation · explain · forecasting ·        │
│      │     inference · registry · pipeline                                              │
│      │                                                                                  │
│      ├── db/ SQLAlchemy 2 + Alembic (SQLite, WAL)                                       │
│      ├── storage/ LocalFileStore: uploads/, artifacts/models/{id}/vN/, outputs/         │
│      └── jobs/ ThreadJobRunner (batch scoring)                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data flow of one run

1. **Upload.** `POST /datasets` streams the file to `storage/uploads/`. The ingest module reads every sheet, finds the header row (skipping title rows and merged cells), cleans the column names and stores a cleaned snapshot (pickle) per sheet. Workbooks with several sheets wait for a sheet choice.
2. **Session.** `POST /sessions` with a goal (and optionally a dataset) creates an `agent_sessions` row and starts an `AgentRun` thread. The thread runs the stages in order.
3. **Events.** Each stage emits typed events (`message.delta`, `tool.*`, `card`, `ask`, `blueprint`, `decision`, `experiment`, `stage`, …). The `EventBus` writes each one to `session_events` with an increasing `seq` and keeps a short in-memory tail. The SSE endpoint replays from `Last-Event-ID` and then follows the tail. See [agent-events.md](agent-events.md).
4. **Questions.** `ask()` blocks the run thread until `POST /sessions/{id}/reply` arrives. Free-text replies are mapped to an option by the LLM, falling back to rules. After each stage the session state is checkpointed as JSON. A run interrupted by a restart is marked retryable; `POST /retry` resumes at the failed stage.
5. **Training.** `ml.pipeline` runs cross-validation, Optuna tuning, the critic and round 2 under a wall-clock budget (`TRAIN_TIME_BUDGET_S`). Every run is reported as an `experiment` event and, when enabled, logged to MLflow.
6. **Publish.** `ml.registry.save_version` writes `model.joblib`, `card.json` and `input_schema.json` to `artifacts/models/{id}/vN/`. It also creates a `model_versions` row and flips the model to `ready`.
7. **Playground.** The model page loads `GET /models/{id}/input-schema` and builds a form from it. `POST /models/{id}/predict` returns predictions with per-row SHAP drivers. `POST /models/{id}/predict-batch` scores an uploaded sheet in the background and offers an `.xlsx` download. Forecasting models get a series and horizon picker instead.

## Boundaries

- `ml/` is a library: it takes DataFrames and config and returns results and progress callbacks. It is fully testable without the API, and `python -m datalyon.cli file.xlsx "goal"` runs it from the terminal.
- `agent/` turns the library into a conversation. It is the only layer that talks to the LLM, and the LLM never sees raw rows beyond a masked sample of at most 5 rows (`agent/privacy.py`).
- The LLM writes words, not numbers. Every metric, threshold and decision comes from code. The LLM phrases explanations, interprets free-text replies and suggests names, and each call validates against a Pydantic schema with a deterministic fallback.
- Statuses are `draft → training → ready | failed`. There is no deploy step, connectors or external serving.

## Errors and logs

Every error leaves the API as `{"error": {"code", "message", "details"}, "request_id"}`. Logs are one JSON object per line and carry `request_id` and `session_id` via context variables.

## Decisions

The ADRs in [adr/](adr/) record the main decisions: monorepo layout, SQLite + local files, SSE over WebSockets, the TypeScript 5.9 pin, the ML engine module layout, and agent runs in threads.
