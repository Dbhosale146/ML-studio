# Build prompt: Datalyon AI Data Scientist (end-to-end POC)

> Paste everything below this line into Claude Code, run from the folder that holds the unzipped `prototype/` directory.

---

## 0. Your role and how to work

You are a senior full-stack ML engineer. Turn the clickable React prototype in `./prototype` (Datalyon · AI Data Scientist) into a **working end-to-end product**. Today the agent runs a fake script with hard-coded numbers. After your work, a user uploads a real Excel file, chats with the agent, a real model is trained and evaluated on their data, and they test that model inside the app.

Working rules:
- Read the whole prototype first (`src/pages/Agent.jsx`, `src/components/agent/Cards.jsx`, `src/pages/ModelDetail.jsx`, `src/data/scenarios.js`, `src/data/models.js`, `src/lib/store.jsx`, `src/components/charts.jsx`). Keep its look, layout, copy style, and UX (chat + live workspace with stepper, Blueprint / Decisions / Experiments tabs, rich cards, model catalog, model detail tabs). Reuse its components wherever you can.
- Build in the milestones in section 11, in order. At the end of each milestone, run lint, type checks and tests, fix failures, then commit with a clear conventional-commit message.
- Do not stop to ask me questions unless you are truly blocked. Make sensible decisions and write each significant one as a short ADR in `docs/adr/`.
- No fake or hard-coded results anywhere in the final product. Every number, chart, decision and explanation must come from the real data and real training runs.

## 1. Scope

In scope for this POC:
- **Data source: file upload only.** Excel (`.xlsx`, `.xls`) is the main format; also accept `.csv`. Remove all warehouse connectors (Snowflake, BigQuery, Postgres, and so on) from the UI and code. Keep the code structured so connectors can be added later behind an interface.
- **Problem types:** binary classification (including imbalanced, fraud-style data), multiclass classification, regression, and time-series forecasting (single or multiple series).
- **The agent builds the model for real:** profiling, problem detection, target selection, strategy, cleaning, feature engineering, model search with HPO, a critic review, a second training round, final evaluation, explanation, and saving a versioned model.
- **Last step = Test the model in the frontend.** A Playground in the app for single predictions (auto-generated form) and batch predictions (upload Excel, download Excel with predictions).

Out of scope for this POC (do not build, but do not block):
- **No public prediction API.** We do not provide a REST/FastAPI endpoint to customers, API keys, curl snippets or "deploy to production". Remove the "Usage & API" tab and the deploy step from the prototype. (The frontend still needs an internal backend to run Python training. That backend is private to the app, not a product feature.)
- Database/warehouse connectors, authentication and multi-tenancy, clustering, unsupervised anomaly detection, model monitoring/drift, cloud deployment.

## 2. Architecture

```
React web app (Vite, TS) ──HTTP + SSE──▶ Internal app server (Python, FastAPI)
                                              │
                         ┌────────────────────┼─────────────────────┐
                         ▼                    ▼                     ▼
                  Agent orchestrator     ML engine (pure       Storage
                  (stage state machine,  Python library:       SQLite metadata DB,
                   LLM for reasoning     profiling, training,  file store for uploads
                   and explanation)      HPO, critic, infer)   + model artifacts, MLflow
```

Key principles:
- **The LLM reasons, the code computes.** The LLM (Gemini) frames the problem, suggests the target, writes the questions and explanations, and narrates the critic findings. All numbers come from deterministic, tested Python code. The LLM never invents a metric.
- **Data privacy:** the LLM only receives the goal, column names/types, summary statistics and at most 5 sample rows, with PII-looking values (emails, phones, names, IDs) masked. It never receives the full dataset.
- **The ML engine is a standalone library** (`datalyon.ml`) with no FastAPI imports, so it can be tested alone and reused later by a worker or CLI.
- **Leakage safety:** every transform is fitted on training folds only, inside one scikit-learn `Pipeline`/`ColumnTransformer` that is saved with the model. The saved artifact takes raw rows in and returns predictions out.
- **Reproducible:** fixed `random_state` (from config), a hash of the dataset saved with every model, and pinned library versions saved in the model card.

## 3. Tech stack

Frontend (`apps/web`):
- Vite + React 19 + **TypeScript** (migrate the prototype from JSX to TSX), Tailwind CSS v4, Recharts, lucide-react, react-router v7.
- TanStack Query for server state; a small typed SSE client for agent events. Replace the `localStorage` store for models with server data; keep `localStorage` only for small UI preferences.
- API types generated from the backend OpenAPI schema with `openapi-typescript` into `src/types/api.gen.ts`.
- Zod for validating SSE event payloads.
- Tooling: ESLint (or keep oxlint) + Prettier, Vitest + React Testing Library, Playwright for E2E.

Backend (`services/api`):
- Python 3.12, **uv** for dependencies and virtualenv, `pyproject.toml`.
- FastAPI + Uvicorn, Pydantic v2, pydantic-settings, SQLAlchemy 2.0 + SQLite, Alembic migrations.
- Data/ML: pandas, openpyxl, xlrd, numpy, scikit-learn, LightGBM, XGBoost, Optuna, SHAP, joblib, MLflow (local tracking, `sqlite` backend), statsmodels only if needed for forecasting diagnostics.
- LLM: Google Gemini via the official `google-genai` SDK, structured JSON output validated by Pydantic. Model name from env var `GEMINI_MODEL` (default to a current Gemini Flash model). Provide a `MockLLM` that works with no API key, using rules and templates, so the whole app runs and tests pass offline.
- Tooling: ruff (lint + format), mypy (strict on `ml/` and `agent/`), pytest + pytest-cov, pre-commit.

Repo-wide: Docker + docker-compose, Makefile, GitHub Actions CI, `.env.example`, `.editorconfig`.

## 4. Folder structure (create exactly this, adjust only with an ADR)

```
datalyon/
├── apps/
│   └── web/
│       ├── public/
│       ├── src/
│       │   ├── app/                  # App.tsx, router.tsx, providers.tsx (QueryClient, theme)
│       │   ├── pages/                # DashboardPage, AgentPage, CatalogPage, ModelDetailPage, DatasetsPage
│       │   ├── features/
│       │   │   ├── agent/            # chat UI, workspace panel, cards/, useAgentSession.ts, events.ts (zod)
│       │   │   ├── datasets/         # upload dropzone, sheet picker, dataset preview table
│       │   │   ├── models/           # catalog, model detail tabs, metric charts
│       │   │   └── playground/       # single-predict form, batch upload, results table
│       │   ├── components/
│       │   │   ├── ui/               # Card, Badge, Button, Tabs, Stat, Empty, CodeBlock... (from prototype ui.jsx)
│       │   │   ├── charts/           # ROC, PR, confusion matrix, residuals, forecast, importance (real data props)
│       │   │   └── layout/           # Layout, Sidebar, Topbar
│       │   ├── lib/                  # apiClient.ts, sse.ts, format.ts, cx.ts
│       │   ├── types/                # api.gen.ts (generated), domain.ts
│       │   ├── styles/index.css
│       │   └── main.tsx
│       ├── tests/                    # unit/ (vitest), e2e/ (playwright)
│       ├── index.html
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts            # dev proxy /api -> http://localhost:8000
│       └── Dockerfile
├── services/
│   └── api/
│       ├── src/datalyon/
│       │   ├── main.py               # FastAPI app factory, CORS, routers, exception handlers
│       │   ├── core/                 # config.py (pydantic-settings), logging.py (structured JSON), errors.py, ids.py
│       │   ├── api/
│       │   │   ├── deps.py
│       │   │   └── routes/           # health.py, datasets.py, sessions.py, models.py, predictions.py
│       │   ├── schemas/              # Pydantic request/response models + agent event models
│       │   ├── db/                   # base.py, session.py, models.py (ORM), repositories/
│       │   ├── storage/              # file_store.py (uploads, artifacts, outputs) behind an interface
│       │   ├── agent/
│       │   │   ├── orchestrator.py   # stage state machine, pause/resume on approvals, cancel
│       │   │   ├── stages/           # detect.py, data.py, strategy.py, prep.py, train.py, critic.py, final.py, publish.py
│       │   │   ├── events.py         # typed event emitter -> SSE + persisted event log
│       │   │   ├── llm/              # base.py (interface), gemini.py, mock.py
│       │   │   ├── prompts/          # versioned prompt templates (.md / .jinja)
│       │   │   └── privacy.py        # builds the LLM-safe data summary, masks PII
│       │   ├── ml/
│       │   │   ├── ingest.py         # Excel/CSV reading and cleaning of the raw sheet
│       │   │   ├── profiling.py      # types, roles, missingness, cardinality, stats, quality score
│       │   │   ├── detection.py      # heuristic problem/target detection (used with or without LLM)
│       │   │   ├── strategy.py       # metric + validation scheme selection
│       │   │   ├── preprocessing.py  # ColumnTransformer builder, cleaning plan
│       │   │   ├── features.py       # date parts, ratios, lags/rolling for forecasting
│       │   │   ├── candidates.py     # model families per problem type + search spaces
│       │   │   ├── training.py       # CV training loop, Optuna HPO, time budget
│       │   │   ├── critic.py         # leakage, overfit, stability, calibration, baseline checks
│       │   │   ├── evaluation.py     # holdout metrics + chart data (ROC/PR points, CM, residuals, forecasts)
│       │   │   ├── explain.py        # permutation importance, SHAP per-prediction drivers
│       │   │   ├── registry.py       # save/load versioned artifacts + model card, MLflow logging
│       │   │   └── inference.py      # schema validation of input rows, predict, batch predict
│       │   └── jobs/                 # runner.py: background job interface + local process-pool implementation
│       ├── alembic/
│       ├── tests/
│       │   ├── unit/                 # ml/ and agent/ logic
│       │   ├── integration/          # API routes, full pipeline on sample files
│       │   └── fixtures/             # small Excel files incl. messy ones
│       ├── pyproject.toml
│       └── Dockerfile
├── data/
│   └── samples/                      # generated demo Excel files (committed, small)
├── scripts/
│   └── generate_samples.py           # builds realistic synthetic sample datasets
├── storage/                          # runtime: uploads/, artifacts/, mlruns/  (gitignored)
├── docs/
│   ├── architecture.md               # diagram + data flow
│   ├── agent-events.md               # SSE event protocol spec
│   ├── ml-pipeline.md                # every stage, rule and threshold
│   └── adr/                          # 0001-record-architecture-decisions.md, ...
├── .github/workflows/ci.yml          # lint, typecheck, tests for web + api
├── docker-compose.yml                # web + api, volume for storage/
├── Makefile                          # setup, dev, test, lint, format, gen-types, samples, e2e
├── .env.example                      # GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER=mock|gemini, limits
├── .pre-commit-config.yaml
├── .editorconfig
├── .gitignore
└── README.md                         # quick start in under 5 commands, screenshots, architecture link
```

Delete `prototype/dist/` and the scripted `scenarios.js` / seed data once the real flow replaces them.

## 5. Internal backend endpoints (`/api/v1`, used only by the web app)

- `GET  /health`
- `POST /datasets` (multipart upload) → dataset id, file name, sheets with row/column counts
- `POST /datasets/{id}/sheet` `{sheet, header_row?}` → confirms which sheet to use
- `GET  /datasets`, `GET /datasets/{id}`, `GET /datasets/{id}/preview?limit=50`, `GET /datasets/{id}/profile`, `DELETE /datasets/{id}`
- `POST /sessions` `{goal, dataset_id?}` → session id (one session = one agent run)
- `GET  /sessions/{id}/events` → **Server-Sent Events** stream; supports `Last-Event-ID` so a page reload replays the history
- `POST /sessions/{id}/reply` `{ask_id, value, text?}` → answers a pending question/approval
- `POST /sessions/{id}/cancel`
- `GET  /models`, `GET /models/{id}`, `GET /models/{id}/versions`, `DELETE /models/{id}`
- `GET  /models/{id}/input-schema` → fields, types, allowed categories, min/max, default values
- `POST /models/{id}/predict` `{rows: [...]}` → predictions + per-row top drivers
- `POST /models/{id}/predict-batch` (Excel/CSV upload) → job id; `GET /predictions/{job_id}/download` → `.xlsx`
- `GET  /activity` → recent real events for the dashboard

Standards: consistent error envelope `{error: {code, message, details}}`, request IDs in logs and responses, upload size and type validation, pagination on list endpoints, OpenAPI tags and examples.

## 6. Agent event protocol (keeps the prototype UI)

The prototype chat already renders `text`, `tool`, `card` and quick-reply messages. Replace the scripted `run()` in `Agent.jsx` with a consumer of these server events. Document them in `docs/agent-events.md` and mirror them as Pydantic models (backend) and Zod schemas (frontend):

- `stage` `{index, key, status}` drives the stepper
- `thinking` `{text}` shows the typing indicator
- `message` `{id, text}` agent text in markdown, streamed in chunks (`message.delta`)
- `tool.start` / `tool.log` / `tool.end` `{id, label, line, duration_ms}` the expandable terminal block, with real log lines from the engine
- `card` `{id, kind, data}` where `kind` ∈ `detection | profile | strategy | cleaning | features | leaderboard | critic | final | published`, and `card.update` for live leaderboard progress
- `ask` `{ask_id, kind: options|free_text|approval|upload|sheet, prompt, options[]}` pauses the run until `/reply`
- `blueprint` `{field, value}` fills the Blueprint tab
- `decision` `{step, decision, reason, alternatives}` fills the Decisions tab
- `experiment` `{run_id, model, round, score, display, best}` fills the Experiments tab
- `model` `{id, status, progress, stage}` updates the catalog and dashboard live
- `error` `{code, message, retryable}` and `done` `{model_id}`

Persist every event in the DB so sessions can be reopened and replayed.

## 7. Agent flow (stage by stage)

The agent asks for approval at the key steps (marked ✋). The user can always type free text instead of clicking an option; the LLM maps it to an option or answers the question.

1. **Goal.** The user types a goal in plain English (keep the suggested-goal chips, but point them at the sample datasets).
2. **Data (upload only).** `ask: upload`. Read the file; if it has several sheets, `ask: sheet` with row counts. Handle real-world Excel problems: header not in row 1, merged header cells, empty trailing rows/columns, Excel error values (`#N/A`, `#DIV/0!`, `#VALUE!`) as missing, numbers stored as text (with thousands separators or currency signs), mixed date formats, Excel serial dates, duplicate column names, fully empty columns. Limits from config (default 50 MB, 500k rows, 300 columns) with a clear message when exceeded.
3. **Profile.** Column types and semantic roles (ID, numeric, categorical, boolean, date/time, free text, constant), missing %, unique counts, distributions, duplicate rows, a 0–100 data-quality score with the reasons. Show the `profile` card.
4. **Problem detection ✋.** Combine heuristics (target cardinality and dtype, presence of a date column plus goal wording like "forecast/next month") with the LLM's reading of the goal and schema. Output: problem type, target column, entity/ID column, time column and series ID (for forecasting), confidence, signals, alternatives considered. If confidence is low, ask a clarifying question (as the prototype does). Ask the user to confirm the target (offer the top alternative targets).
5. **Strategy.** Pick the metric and validation from rules, then let the LLM explain the choice:
   - Binary: ROC-AUC primary; PR-AUC when positives < 10%; F1, precision, recall, Brier also reported. Stratified 5-fold CV + 20% stratified holdout, or a time-based holdout when a date column exists.
   - Multiclass: macro-F1 primary, accuracy, per-class report. Stratified K-fold.
   - Regression: RMSE or MAE primary (MAE when the target is heavily skewed), R², MAPE when the target is always > 0. K-fold.
   - Forecasting: ask the horizon ✋ (and frequency if it can't be inferred). Rolling-origin backtest (e.g. 3–5 windows of the horizon length). MAPE/WAPE primary, MAE reported. Seasonal-naive baseline.
   - Imbalance handling (class weights), CV folds and time budget come from data size.
6. **Cleaning plan ✋.** Propose concrete fixes with counts (drop ID/constant/duplicate columns, de-duplicate rows, impute with missing-indicator flags, cap outliers at percentiles, normalise category labels such as `"Pro"`, `"pro "`, `"PRO"`, group rare categories, parse dates). Show the `cleaning` card, then ask to approve or skip items.
7. **Features.** Date parts, cyclical encodings, simple ratios/interactions the LLM suggests from column names (validated by code: only accepted if they parse and improve CV), and lags/rolling stats for forecasting (all lags ≥ horizon so nothing leaks). Show the `features` card, marking which features were LLM-suggested.
8. **Model search, round 1.** Always include a dummy baseline. Classification: Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost. Regression: Ridge, Random Forest, HistGradientBoosting, LightGBM, XGBoost. Forecasting: seasonal naive, global LightGBM and XGBoost on lag features. Shortlist by data size (skip slow models on big data, explain why). Run Optuna HPO within the time budget (default 5 minutes total, configurable), run in the background job runner, and stream real progress to the `leaderboard` card and Experiments tab. Log every run to MLflow.
9. **Critic review.** Automatic checks, each with severity and evidence:
   - Target leakage: any single feature with near-perfect predictive power (e.g. single-feature AUC > 0.97 or |corr| > 0.95 with the target), plus columns the LLM flags as "known only after the outcome" by name.
   - Overfitting: train vs validation gap above a threshold.
   - Stability: high standard deviation across folds.
   - Calibration (binary): reliability curve; apply isotonic or Platt calibration if needed.
   - Must beat the baseline; if no model does, say so honestly and stop with advice.
   - Residual patterns (regression/forecasting): error by segment or by period.
   The LLM writes the narrative; the code produces the findings. Show the `critic` card.
10. **Round 2.** Re-train the shortlist with the critic's fixes (removed leaky features, tuned objective, calibration). Show round-1 vs round-2 scores.
11. **Final evaluation ✋.** Choose the best model by holdout score with stability and speed as tie-breakers. Show the `final` card: primary metric vs baseline, full metric table, real chart data (ROC and PR points, confusion matrix at the chosen threshold, actual vs predicted, residual histogram, backtest and forecast plot), permutation feature importance, a plain-English explanation and "why this model" text written by the LLM from the metrics JSON. The user accepts or asks for changes.
12. **Publish for testing (replaces Deploy).** Save the fitted pipeline (joblib), the input schema, the model card JSON (data hash, columns, metrics, decisions, library versions, training time) as version `vN` under `storage/artifacts/models/{model_id}/`. Set status `ready`. Show a `published` card with a **"Test this model"** button that opens the Playground.

Model statuses in the UI: `draft → training → ready`, plus `failed`. Remove `live/staging`.

## 8. Frontend pages

- **Dashboard:** real KPIs (models, ready models, datasets, predictions run in the Playground), model status list, recent activity, "Create new model" goal box. Replace the connectors panel with a **Datasets** panel (recent uploads + upload button). Proper empty states when nothing exists yet.
- **Create New Model (Agent):** same chat + live workspace as the prototype, now driven by SSE. Drag-and-drop Excel into the chat. Cancel button. Reconnects and replays on page reload. Errors appear in the chat with a Retry button.
- **Datasets:** list, preview table (first 50 rows), profile view, delete.
- **Model Catalog:** search and filter by problem type and status (real data).
- **Model detail tabs:** Overview, Data, Metrics, Decision log, Versions, and **Test model** (replaces "Usage & API"):
  - Single prediction: form generated from `input-schema` (number inputs with min/max hints, dropdowns for categories, date pickers), prefilled with typical values, "Fill random row from training data" button. Result: predicted label/value, probability or interval, risk band for classification, top 5 drivers (SHAP) as bars.
  - Batch: upload Excel/CSV with the same columns, validate and report missing/extra columns, show the first rows of results, download `.xlsx` with prediction columns added.
  - Forecasting models: pick a series and horizon, show the forecast chart with intervals, download the forecast.
- Remove the user name hard-coded in the agent greeting ("Akash"); read the display name from config.
- Accessibility: keyboard navigation, focus states, aria labels on icon buttons. Responsive down to tablet width.

## 9. Sample data

`scripts/generate_samples.py` creates realistic synthetic Excel files in `data/samples/` with known signal and some deliberate mess (missing values, inconsistent labels, a leaky column, duplicate rows, a second sheet):
- `customer_churn.xlsx` (binary, ~12% positive, includes a leaky `cancellation_reason` column)
- `card_fraud.xlsx` (binary, ~1% positive)
- `house_prices.xlsx` (regression)
- `store_weekly_demand.xlsx` (forecasting, ~10 stores × 3 years weekly)
- `product_category.xlsx` (multiclass)

## 10. Quality bar (non-functional)

- Tests: unit tests for every `ml/` module and the orchestrator (with `MockLLM`), integration test that runs the full pipeline on each sample file and asserts the model beats its baseline and the critic catches the leaky churn column. Backend coverage ≥ 80% on `ml/` and `agent/`. Frontend unit tests for the event reducer and Playground form; one Playwright E2E: upload churn sample → finish agent flow → single prediction → batch download.
- Logging: structured JSON logs with request and session IDs; no raw data rows in logs.
- Config: everything tunable via env (`pydantic-settings`), documented in `.env.example`.
- Errors: user-friendly messages in chat, full details in logs; the run can be retried from the failed stage.
- Security basics even for a POC: validate file type by content, not only extension; size limits; no `eval`; the LLM output is always parsed through a schema, never executed; secrets only from env.
- Performance: a 50k-row file finishes the whole flow within the default time budget on a laptop; the UI stays responsive during training (work runs in the background job runner, not the request thread).
- Docs: README quick start, `docs/architecture.md`, `docs/ml-pipeline.md`, `docs/agent-events.md`, ADRs.

## 11. Milestones (build in this order, verify each)

1. **Repo setup:** monorepo layout, move the prototype into `apps/web`, migrate to TypeScript, tooling, Makefile, docker-compose, CI, pre-commit. App still runs with the old script.
2. **Backend skeleton:** FastAPI app, config, logging, DB + Alembic, storage, health route, OpenAPI → generated TS types.
3. **Ingest + profiling:** upload, sheet selection, messy-Excel handling, profile endpoint, Datasets page. Sample generator script.
4. **ML engine:** detection, strategy, preprocessing, features, candidates, training + HPO, critic, evaluation, explain, registry, inference, all as a library with unit tests, runnable from a CLI script on the sample files.
5. **Agent orchestrator + LLM:** state machine, approvals, event emitter, SSE, persisted events, `MockLLM` and Gemini provider, prompts, privacy summary.
6. **Frontend agent wiring:** replace the scripted conversation with the SSE consumer; all cards render real data; workspace tabs fill from events; catalog and dashboard read from the API.
7. **Playground:** input schema, single and batch prediction, forecasting view, Test model tab.
8. **Hardening:** error and retry paths, cancel, reload/replay, empty states, accessibility pass, E2E test, docs, README with screenshots.

## 12. Definition of done

- `make setup && make dev` starts web on `http://localhost:5173` and the backend on `:8000`; `docker compose up` does the same.
- With `LLM_PROVIDER=mock` (no key) and with `LLM_PROVIDER=gemini`, I can take every sample file from upload to a `ready` model and test it in the Playground.
- No scripted scenarios, seed models, fake connectors, fake metrics or "Usage & API" / deploy UI remain.
- `make lint`, `make typecheck`, `make test` and `make e2e` all pass, and CI is green.
- Finish by printing a short summary: what was built, how to run it, known limitations, and suggested next steps (connectors, REST serving, monitoring, auth).
