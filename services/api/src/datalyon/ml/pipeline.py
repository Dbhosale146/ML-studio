"""Stage functions that compose the engine. The agent calls them one by one (with approvals in between);
run_all() chains them with automatic approvals for the CLI and integration tests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import forecasting as fc
from .candidates import Candidate, classification_candidates, regression_candidates
from .critic import apply_fixes, run_critic
from .detection import detect
from .evaluation import FinalResult, choose_final, fit_final
from .features import rule_based_derived
from .inference import build_input_schema, example_rows, reference_row
from .metrics import LABELS, display, forecast_metrics, metric_table
from .preprocessing import build_cleaning_plan, clean_rows, config_from_steps
from .profiling import profile_dataframe
from .registry import Tracker, library_versions
from .strategy import FREQ_INFO, choose_strategy, horizon_from_goal, infer_frequency
from .training import TabularTask, evaluate_candidate, prepare_task, tune
from .types import Detection, Finding, Progress, RunResult, Strategy, TrainConfig, noop_progress

log = logging.getLogger("datalyon.ml.pipeline")
MAX_TRIALS = 25


def candidates_for(task: TabularTask, n_jobs: int) -> tuple[dict[str, Candidate], list[dict[str, str]]]:
    n = len(task.y_train) + len(task.y_hold)
    if task.problem_type == "regression":
        cands, skipped = regression_candidates(n, task.seed, n_jobs)
    else:
        cands, skipped = classification_candidates(n, task.n_classes, task.seed, n_jobs)
    return {c.key: c for c in cands}, skipped


def _log_run(tracker: Tracker | None, run: RunResult, metric: str) -> None:
    if tracker:
        tracker.log_run(
            run.run_id,
            {
                "candidate": run.candidate,
                "round": run.round,
                **run.params,
                **{f"cfg_{k}": v for k, v in run.config.items() if k != "derived"},
            },
            {metric: run.mean, f"{metric}_std": run.std},
        )


def supervised_round(
    task: TabularTask,
    cands: dict[str, Candidate],
    cfg: TrainConfig,
    round_: int,
    deadline: float,
    progress: Progress = noop_progress,
    tracker: Tracker | None = None,
    params_by_key: dict[str, dict[str, Any]] | None = None,
    only: list[str] | None = None,
    tune_top: int = 2,
) -> list[RunResult]:
    """CV every (selected) candidate, then Optuna HPO on the best `tune_top` within the deadline."""
    runs: list[RunResult] = []
    keys = only or list(cands)
    for key in keys:
        cand = cands[key]
        has_model = any(r.family != "Baseline" for r in runs)
        if time.perf_counter() > deadline and not cand.baseline and has_model:  # always train at least one real model
            progress("skip", {"candidate": cand.name, "key": key, "round": round_, "reason": "time budget reached"})
            continue
        params = (params_by_key or {}).get(key, {})
        progress("start", {"candidate": cand.name, "key": key, "round": round_})
        try:
            run = evaluate_candidate(task, cand, round_, params, cfg, progress)
        except Exception as e:
            progress("failed", {"candidate": cand.name, "key": key, "round": round_, "error": str(e)[:200]})
            continue
        runs.append(run)
        _log_run(tracker, run, task.strategy.metric)
        progress("run", {"run": run.summary(), "key": key})
    ranked = sorted((r for r in runs if r.family != "Baseline" and np.isfinite(r.mean)), key=lambda r: -r.selection_score)
    key_of = {cands[k].name: k for k in keys}
    tunable = [key_of[r.candidate] for r in ranked if cands[key_of[r.candidate]].suggest is not None][:tune_top]
    for i, key in enumerate(tunable):
        left = deadline - time.perf_counter()
        budget = left / (len(tunable) - i) * 0.85
        if budget < 3:
            break
        cand = cands[key]
        progress("tune_start", {"candidate": cand.name, "key": key, "round": round_, "budget_s": round(budget, 1)})
        best_params, _, trials = tune(task, cand, cfg, budget, MAX_TRIALS, progress)
        if not best_params:
            continue
        merged = {**(params_by_key or {}).get(key, {}), **best_params}
        tuned = evaluate_candidate(task, cand, round_, merged, cfg, progress, tuned=True)
        _log_run(tracker, tuned, task.strategy.metric)
        prev = next(r for r in runs if r.candidate == cand.name)
        progress("tuned", {"run": tuned.summary(), "key": key, "trials": trials, "improved": tuned.selection_score > prev.selection_score})
        runs.append(tuned)
    return runs


def best_params(runs: list[RunResult], cands: dict[str, Candidate]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, cand in cands.items():
        mine = [r for r in runs if r.candidate == cand.name and np.isfinite(r.mean)]
        if mine:
            out[key] = max(mine, key=lambda r: r.selection_score).params
    return out


def top_keys(runs: list[RunResult], cands: dict[str, Candidate], k: int) -> list[str]:
    best: dict[str, float] = {}
    for r in runs:
        if r.family != "Baseline" and np.isfinite(r.mean):
            best[r.candidate] = max(best.get(r.candidate, float("-inf")), r.selection_score)
    names = sorted(best, key=lambda n: -best[n])[:k]
    return [key for key, c in cands.items() if c.baseline] + [key for n in names for key, c in cands.items() if c.name == n]


def round2_keys(runs1: list[RunResult], cands: dict[str, Candidate], cfg: TrainConfig, cfg2: TrainConfig) -> list[str]:
    """After a leakage fix every round-1 score is inflated, so all candidates are re-run; otherwise the top 3."""
    if set(cfg2.exclude) != set(cfg.exclude):
        return list(cands)
    return top_keys(runs1, cands, 3)


def build_tabular_artifact(
    task: TabularTask,
    final: FinalResult,
    cands: dict[str, Candidate],
    df_clean: pd.DataFrame,
    labels: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Refit the chosen pipeline on every row and bundle what inference needs."""
    t0 = time.perf_counter()
    cand = cands[final.candidate_key]
    model = fit_final(task, cand, final.params, final.config, task.X_all(), task.y_all())
    secs = time.perf_counter() - t0
    columns = [c for c in task.columns if c not in final.config.exclude]
    importance = {str(c): float(v) for c, v in final.importance}
    schema = build_input_schema(df_clean, columns, task.kinds, importance, labels)
    X_all = task.X_all()
    artifact = {
        "kind": "tabular",
        "problem_type": task.problem_type,
        "target": task.target,
        "columns": columns,
        "kinds": {c: task.kinds[c] for c in columns},
        "classes": task.classes,
        "pos_index": task.pos_index,
        "threshold": final.threshold,
        "model": model,
        "reference": reference_row(X_all, columns, task.kinds),
        "background": X_all[columns].sample(min(100, len(X_all)), random_state=task.seed),
        "residual_q": final.charts.get("residual_quantiles"),
        "input_schema": schema,
        "examples": example_rows(df_clean, columns, task.target),
    }
    return artifact, schema, secs


def make_card(
    *,
    model_name: str,
    problem_type: str,
    goal: str,
    target: str | None,
    algorithm: str,
    params: dict[str, Any],
    config: dict[str, Any],
    strategy: Strategy,
    metrics: list[dict[str, Any]],
    data_hash: str,
    dataset: dict[str, Any],
    columns: list[str],
    excluded: list[dict[str, Any]],
    training_seconds: float,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": model_name,
        "problem_type": problem_type,
        "goal": goal,
        "target": target,
        "algorithm": algorithm,
        "params": params,
        "config": config,
        "strategy": strategy.to_dict(),
        "metrics": metrics,
        "data_hash": data_hash,
        "dataset": dataset,
        "columns": columns,
        "excluded": excluded,
        "training_seconds": round(training_seconds, 1),
        "random_state": seed,
        "library_versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(),
        **(extra or {}),
    }


# ---------------------------------------------------------------------------------------------
# Straight-through runner (CLI / tests): every approval is accepted with its default.
# ---------------------------------------------------------------------------------------------
@dataclass
class PipelineResult:
    detection: Detection
    strategy: Strategy
    findings: list[Finding]
    runs: list[RunResult]
    metrics: list[dict[str, Any]]
    algorithm: str
    artifact: dict[str, Any]
    schema: list[dict[str, Any]]
    excluded: list[str] = field(default_factory=list)
    stopped: str | None = None


def run_all(
    df: pd.DataFrame,
    goal: str,
    budget_s: int = 60,
    seed: int = 42,
    n_jobs: int = 2,
    progress: Progress = noop_progress,
    tracker: Tracker | None = None,
) -> PipelineResult:
    profile = profile_dataframe(df)
    det = detect(goal, profile, df)
    if det.target is None:
        raise ValueError("Could not detect a target column from the goal.")
    start = time.perf_counter()
    if det.problem_type == "forecasting":
        return _run_forecast(df, goal, det, budget_s, seed, n_jobs, progress, start)
    strategy = choose_strategy(det.problem_type, df, det.target, det.time_column, budget_s, det.positive_class)
    steps = build_cleaning_plan(df, profile, det.target, [])
    cfg = config_from_steps(steps, TrainConfig(class_weight=False))
    clean, _ = clean_rows(df, det.target, dedupe=True)
    task = prepare_task(clean, profile, det.problem_type, det.target, strategy, [], det.time_column, det.positive_class, seed)
    cfg.derived = rule_based_derived(task.columns, task.kinds, profile["columns"], det.target)
    cands, _ = candidates_for(task, n_jobs)
    deadline1 = start + budget_s * 0.55
    runs1 = supervised_round(task, cands, cfg, 1, deadline1, progress, tracker)
    findings = run_critic(task, runs1, cands, cfg, det.leakage_suspects)
    stop = next((f for f in findings if f.fix and f.fix.get("stop")), None)
    cfg2, _ = apply_fixes(cfg, findings)
    keys = round2_keys(runs1, cands, cfg, cfg2)
    runs2 = supervised_round(task, cands, cfg2, 2, start + budget_s * 0.9, progress, tracker, best_params(runs1, cands), keys, tune_top=1)
    final = choose_final(task, runs2 or runs1, cands, cfg2)
    artifact, schema, _ = build_tabular_artifact(task, final, cands, clean, {})
    return PipelineResult(
        det, strategy, findings, runs1 + runs2, final.metrics, final.candidate, artifact, schema, cfg2.exclude, stop.title if stop else None
    )


def _run_forecast(
    df: pd.DataFrame, goal: str, det: Detection, budget_s: int, seed: int, n_jobs: int, progress: Progress, start: float
) -> PipelineResult:
    assert det.target and det.time_column
    freq = infer_frequency(df[det.time_column]) or "D"
    horizon = horizon_from_goal(goal, freq) or int(FREQ_INFO[freq]["default_h"])
    strategy = choose_strategy("forecasting", df, det.target, det.time_column, budget_s, horizon=horizon, freq=freq)
    task = fc.prepare_forecast(df, det.target, det.time_column, det.series_column, freq, horizon, strategy.metric, seed)
    cfg = fc.default_config(task)
    windows = fc.origins(task)
    runs, preds = forecast_round(task, cfg, windows, 1, start + budget_s * 0.55, n_jobs, progress)
    findings = fc.critic(task, runs, preds, cfg)
    cfg2, _ = fc.apply_fixes(cfg, findings)
    runs2, preds2 = forecast_round(task, cfg2, windows, 2, start + budget_s * 0.9, n_jobs, progress, params=fc_best_params(runs))
    artifact, metrics, algorithm = build_forecast_artifact(task, runs + runs2, {**preds, **preds2}, n_jobs)
    stop = next((f for f in findings if f.fix and f.fix.get("stop")), None)
    return PipelineResult(det, strategy, findings, runs + runs2, metrics, algorithm, artifact, [], [], stop.title if stop else None)


# ---------------------------------------------------------------------------------------------
# Forecasting stages
# ---------------------------------------------------------------------------------------------
def forecast_round(
    task: fc.ForecastTask,
    cfg: fc.FcConfig,
    windows: list[pd.Timestamp],
    round_: int,
    deadline: float,
    n_jobs: int,
    progress: Progress = noop_progress,
    tracker: Tracker | None = None,
    params: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[RunResult], dict[str, pd.DataFrame]]:
    runs: list[RunResult] = []
    preds: dict[str, pd.DataFrame] = {}
    cands = fc.forecast_candidates()
    for cand in cands:
        if time.perf_counter() > deadline and not cand.baseline and any(r.family != "Baseline" for r in runs):
            progress("skip", {"candidate": cand.name, "key": cand.key, "round": round_, "reason": "time budget reached"})
            continue
        progress("start", {"candidate": cand.name, "key": cand.key, "round": round_})
        run, p = fc.evaluate(task, cand, (params or {}).get(cand.key, {}), cfg, windows, round_, n_jobs, progress)
        runs.append(run)
        preds[run.run_id] = p
        _log_run(tracker, run, task.metric)
        progress("run", {"run": run.summary(), "key": cand.key})
    ranked = sorted((r for r in runs if r.family != "Baseline"), key=lambda r: -r.selection_score)
    if ranked:
        top = next(c for c in cands if c.name == ranked[0].candidate)
        budget = (deadline - time.perf_counter()) * 0.85
        if budget >= 3:
            progress("tune_start", {"candidate": top.name, "key": top.key, "round": round_, "budget_s": round(budget, 1)})
            bp, _, trials = fc.tune(task, top, cfg, windows, budget, MAX_TRIALS, n_jobs, progress)
            if bp:
                run, p = fc.evaluate(task, top, bp, cfg, windows, round_, n_jobs, progress, tuned=True)
                runs.append(run)
                preds[run.run_id] = p
                _log_run(tracker, run, task.metric)
                progress(
                    "tuned",
                    {"run": run.summary(), "key": top.key, "trials": trials, "improved": run.selection_score > ranked[0].selection_score},
                )
    return runs, preds


def fc_best_params(runs: list[RunResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in fc.forecast_candidates():
        mine = [r for r in runs if r.candidate == c.name]
        if mine:
            out[c.key] = max(mine, key=lambda r: r.selection_score).params
    return out


def build_forecast_artifact(
    task: fc.ForecastTask, runs: list[RunResult], preds: dict[str, pd.DataFrame], n_jobs: int
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Pick the best run on the backtest, refit on all history, forecast the horizon and package the chart data."""
    cands = {c.name: c for c in fc.forecast_candidates()}
    best = max((r for r in runs if np.isfinite(r.mean)), key=lambda r: (r.selection_score, -r.std))
    cand = cands[best.candidate]
    cfg = fc.FcConfig(**best.config)
    p = preds[best.run_id]
    q = fc.interval_quantiles(task, p)
    model = fc.fit_full(task, cand, best.params, cfg, n_jobs)
    future = fc.forecast_future(task, cand.key, model, cfg, q)
    last_window = p[p["window"] == p["window"].max()]
    base = next((r for r in runs if r.family == "Baseline"), None)
    model_m = forecast_metrics(p["y"].to_numpy(), p["yhat"].to_numpy())
    base_m = forecast_metrics(preds[base.run_id]["y"].to_numpy(), preds[base.run_id]["yhat"].to_numpy()) if base else {}
    model_m["coverage_80"] = fc.coverage(p, q)
    metrics = metric_table(model_m, base_m, [task.metric, "wape" if task.metric == "mape" else "mape", "mae", "rmse", "coverage_80"])
    windows = sorted(p["window"].unique())
    per_window = []
    for w in windows:
        a = p[p["window"] == w]
        row = {
            "window": f"W{w}",
            "start": str(a["ds"].min().date()),
            task.metric: forecast_metrics(a["y"].to_numpy(), a["yhat"].to_numpy())[task.metric],
        }
        if base:
            b = preds[base.run_id]
            b = b[b["window"] == w]
            row["baseline"] = forecast_metrics(b["y"].to_numpy(), b["yhat"].to_numpy())[task.metric]
        per_window.append(row)
    chart = fc.chart_data(task, future, last_window)
    artifact = {
        "kind": "forecast",
        "problem_type": "forecasting",
        "target": task.target,
        "model": model,
        "candidate": cand.key,
        "forecast": {
            "horizon": task.horizon,
            "unit": task.unit,
            "freq": task.freq,
            "series": task.series,
            "series_column": task.series_column,
            "chart": chart,
            "intervals": q,
            "config": cfg.to_dict(),
            "params": best.params,
            "backtest": per_window,
            "metric": task.metric,
            "metric_label": LABELS[task.metric],
            "cv_mean": best.mean,
            "display": display(task.metric, best.mean),
        },
        "input_schema": [],
    }
    return artifact, metrics, cand.name
