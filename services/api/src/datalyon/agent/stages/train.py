"""Stage 5: model search round 1 (CV of every shortlisted family + Optuna HPO on the leaders)."""

from __future__ import annotations

import time

from ...ml import forecasting as fc
from ...ml.candidates import shortlist_reason
from ...ml.metrics import LABELS, display
from ...ml.pipeline import forecast_round, supervised_round
from ..context import StageContext
from .common import Leaderboard, ensure_candidates, ensure_fc_task, ensure_task, tracker, train_config


def run(ctx: StageContext) -> None:
    plan = ctx.state["plan"]
    budget = ctx.settings.train_time_budget_s
    ctx.state["train_started"] = time.time()
    ctx.model_update(status="training", stage="Model search, round 1", progress=20)
    if plan["problem_type"] == "forecasting":
        _forecast(ctx, budget)
        return
    task = ensure_task(ctx)
    cands = ensure_candidates(ctx)
    cfg = train_config(ctx)
    n_cat = sum(1 for c in task.columns if task.kinds[c] == "category")
    ctx.say(
        shortlist_reason(plan["problem_type"], len(task.y_train) + len(task.y_hold), len(task.columns), n_cat, ctx.mem.get("skipped", []))
    )
    ctx.say(
        f"I'll compare them with {task.strategy.validation[0].lower() + task.strategy.validation[1:]}, then spend part of the {_duration(budget)} budget tuning the two best with Optuna."
    )
    ctx.decision(
        "Model search",
        f"Shortlisted {len(cands) - 1} model families + a baseline",
        "Chosen from data size and column types.",
        "; ".join(f"{s['name']} ({s['reason']})" for s in ctx.mem.get("skipped", [])),
    )
    board = Leaderboard(ctx, 1, [(k, c.name, c.family) for k, c in cands.items()], task.strategy.metric, progress_span=(20, 55))
    runs = supervised_round(task, cands, cfg, 1, time.perf_counter() + budget * 0.55, board.progress, tracker(ctx))
    best = board.finish()
    ctx.mem["runs1"] = runs
    ctx.state["round1"] = [r.summary() for r in runs]
    if best:
        ctx.blueprint("Round 1 leader", f"{best['name']} · {LABELS[task.strategy.metric]} {best['display']}")
        ctx.say(
            f"Round 1 is done. **{best['name']}** leads with {LABELS[task.strategy.metric]} **{best['display']}**. Before I accept that, the critic reviews the results."
        )


def _duration(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s}s" if m and s else f"{m} min" if m else f"{s}s"


def _forecast(ctx: StageContext, budget: int) -> None:
    task = ensure_fc_task(ctx)
    cfg = fc.FcConfig(**ctx.state["fc_cfg"])
    windows = fc.origins(task)
    ctx.mem["windows"] = windows
    cands = fc.forecast_candidates()
    ctx.say(
        f"I'll backtest a seasonal-naive baseline against global gradient-boosting models (one model across all {len(task.series)} series) "
        f"on {len(windows)} rolling windows of {task.horizon} {task.unit}s, then tune the best with Optuna."
    )
    ctx.decision(
        "Model search",
        "Seasonal naive vs global LightGBM / XGBoost",
        "Global models share patterns across series; seasonal naive is the bar to beat.",
    )
    board = Leaderboard(ctx, 1, [(c.key, c.name, c.family) for c in cands], task.metric, progress_span=(20, 55))
    runs, preds = forecast_round(task, cfg, windows, 1, time.perf_counter() + budget * 0.55, ctx.n_jobs, board.progress, tracker(ctx))
    best = board.finish()
    ctx.mem["runs1"], ctx.mem["preds"] = runs, preds
    ctx.state["round1"] = [r.summary() for r in runs]
    if best:
        ctx.say(
            f"Round 1 is done. **{best['name']}** leads with {LABELS[task.metric]} **{display(task.metric, best['score'])}** across the backtest windows. The critic reviews it next."
        )
