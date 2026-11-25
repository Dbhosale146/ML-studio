# Agent event protocol

One **session** is one agent run. The web app starts it with `POST /api/v1/sessions {goal, dataset_id?}` and then
listens to `GET /api/v1/sessions/{id}/events`, a **Server-Sent Events** stream.

- Every SSE message has `id: <seq>` and `data: {"seq", "type", "at", "payload"}`.
- Every event is validated by a Pydantic model (`services/api/src/datalyon/schemas/events.py`) before it is stored in
  the `session_events` table and streamed. The frontend validates it again with Zod (`apps/web/src/features/agent/events.ts`).
- Reconnects send `Last-Event-ID` (the browser does this automatically) and receive only newer events. Opening a
  finished session replays the whole conversation from the database.
- When a run is finished (`done`, `failed` or `cancelled`) and everything was sent, the server sends `event: end` and
  closes the stream; the client then closes its `EventSource`.

## Commands

| Endpoint | Purpose |
|---|---|
| `POST /sessions/{id}/reply {ask_id, value, text?}` | Answer the pending `ask` (a click sends `value`; typing sends `text`, which the LLM maps to an option or answers as a question before asking again). |
| `POST /sessions/{id}/message {text}` | Free text at any time: answers the pending question if there is one, otherwise the agent answers it from the run's facts. |
| `POST /sessions/{id}/cancel` | Cancel; training stops at the next fold/trial. |
| `POST /sessions/{id}/retry` | Retry a failed or cancelled run from the stage where it stopped (state is checkpointed after every stage). |

## Events

| Type | Payload | UI |
|---|---|---|
| `stage` | `{index, key, label, status: active\|done\|failed\|skipped}` | Stepper. Keys: `data, detect, strategy, prep, train, critic, final, publish`. |
| `thinking` | `{text}` (`null` clears it) | Typing indicator with a status line. |
| `message` | `{id, text}` | Agent message (markdown: `**bold**`, `` `code` ``). Long messages start with `text: ""`… |
| `message.delta` | `{id, delta}` | …and stream in chunks, appended in order. |
| `tool.start` / `tool.log` / `tool.end` | `{id, label}` / `{id, line}` / `{id, duration_ms, status}` | Expandable terminal block with real log lines from the engine. |
| `card` | `{id, kind, data}` | Rich card. `kind` ∈ `profile, detection, strategy, cleaning, features, leaderboard, critic, final, published`. |
| `card.update` | `{id, data}` | Replaces a card's data (live leaderboard progress, approved cleaning plan, switched final model). |
| `ask` | `{ask_id, kind, prompt, options[], items[], default}` | Pauses the run. `kind`: `options` (quick replies), `approval` (checklist in `items`, value = list of enabled optional ids), `upload` (value = dataset id from `POST /datasets`), `sheet` (value = sheet name), `free_text`. |
| `reply` | `{ask_id, text, value}` | The user's answer, echoed so replays show both sides. |
| `blueprint` | `{field, value}` | Blueprint tab. |
| `decision` | `{step, decision, reason, alternatives}` | Decisions tab (also saved in the model card). |
| `experiment` | `{run_id, model, round, score, display, best, tuned}` | Experiments tab (also logged to MLflow). |
| `model` | `{id, status, progress, stage, name}` | Live status for the catalog and dashboard (`draft → training → ready`, or `failed`). |
| `session` | `{status: running\|waiting\|done\|failed\|cancelled}` | Composer state. |
| `error` | `{code, message, retryable}` | Error bubble with a Retry button when `retryable`. `no_signal` (no model beats the baseline) is not retryable. |
| `done` | `{model_id}` | Run finished; the `published` card links to the Test tab. |

## Card data (main fields)

- **profile**: `file_name, sheet, rows, cols, quality, quality_reasons[], period, columns[{name, kind, role, missing_pct, unique, pii}], issues[], notes[], preview{columns, rows}`
- **detection**: `problem_type, problem_label, confidence, target, time_column, series_column, entity_column, positive_class, signals[], alternatives[], candidate_targets[]`
- **strategy**: `metric_label, better, secondary[], validation, explanation, reasons[], extras[[k, v]], time_budget_s`
- **cleaning**: `items[{id, title, detail, count, optional, enabled, columns}], approved`
- **features**: `items[{name, kind: raw|date|missing|derived|lag|rolling|calendar, desc, source: llm|rules, accepted, gain}], total, inputs`
- **leaderboard**: `round, metric_label, better, rows[{key, name, family, status, progress, display, std, tuned, prev_display, interim, note}], trials, tuning, done`
- **critic**: `headline, findings[{id, check, severity, title, detail, evidence, fix}], applied[]`
- **final**: `algorithm, primary{name, display, baseline}, metrics[{name, display, baseline, good}], charts{roc, pr, confusion, calibration, scores, per_class, scatter, residuals, forecast, backtest}, importance[[feature, share]], explanation, why_model, ranking[], threshold`
- **published**: `model_id, version, name, algorithm, primary, test_url`
