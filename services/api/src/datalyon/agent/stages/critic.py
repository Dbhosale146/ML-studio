"""Stage 6: critic review of round 1, fixes, and round 2."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...ml import forecasting as fc
from ...ml.critic import apply_fixes, run_critic
from ...ml.metrics import LABELS, LOWER_IS_BETTER, display
from ...ml.pipeline import best_params, fc_best_params, forecast_round, round2_keys, supervised_round
from ...ml.types import Finding, RunResult
from ..context import StageContext, StopRunError
from .common import Leaderboard, ResumeFromError, ensure_candidates, ensure_fc_task, ensure_task, tracker, train_config

SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3, "pass": 4}


def _leader(runs: list[RunResult]) -> RunResult | None:
    models = [r for r in runs if r.family != "Baseline" and np.isfinite(r.mean)]
    return max(models, key=lambda r: r.selection_score) if models else None


def _report(ctx: StageContext, findings: list[Finding], leader: RunResult, metric: str, applied: list[Finding]) -> None:
    facts = {
        "leader": leader.candidate,
        "metric": LABELS[metric],
        "score": display(metric, leader.mean),
        "findings": [{"check": f.check, "severity": f.severity, "title": f.title, "detail": f.detail} for f in findings],
        "fixes_for_round_2": [f.title for f in applied],
    }
    with ctx.tool("critic.narrate(findings)"):
        headline = ctx.llm.narrate_critic(facts)
    ordered = sorted(findings, key=lambda f: SEV_ORDER[f.severity])
    ctx.card(
        "critic",
        {
            "headline": headline,
            "findings": [f.to_dict() for f in ordered],
            "applied": [f.id for f in applied],
            "leader": leader.candidate,
            "score": display(metric, leader.mean),
        },
    )
    ctx.state["critic"] = {"headline": headline, "findings": [f.to_dict() for f in ordered], "applied": [f.id for f in applied]}
    for f in applied:
        ctx.decision("Critic", f.title, f.detail)


def run(ctx: StageContext) -> None:
    if "runs1" not in ctx.mem:
        raise ResumeFromError("train")
    plan = ctx.state["plan"]
    budget = ctx.settings.train_time_budget_s
    ctx.model_update(stage="Critic review", progress=58)
    if plan["problem_type"] == "forecasting":
        _forecast(ctx, budget)
        return
    task, cands, cfg = ensure_task(ctx), ensure_candidates(ctx), train_config(ctx)
    runs1: list[RunResult] = ctx.mem["runs1"]
    metric = task.strategy.metric
    suspects = plan.get("leakage_suspects", [])
    with ctx.tool("critic.review(round=1)") as t:
        findings = run_critic(task, runs1, cands, cfg, suspects)
        for f in findings:
            t.log(f"{f.severity:>6}  {f.title}")
    stop = next((f for f in findings if f.fix and f.fix.get("stop")), None)
    cfg2, applied = apply_fixes(cfg, findings)
    leader = _leader(runs1)
    assert leader is not None
    _report(ctx, findings, leader, metric, applied)
    if stop:
        raise StopRunError(
            "no_signal",
            f"{stop.detail} Suggestions: check that `{plan['target']}` is recorded correctly, add columns that plausibly influence it, or collect more rows, then start a new run.",
        )
    ctx.state["cfg2"] = cfg2.to_dict()
    keys = round2_keys(runs1, cands, cfg, cfg2)
    changed = cfg2.to_dict() != cfg.to_dict()
    ctx.say(
        (
            "Applying the fixes and retraining "
            + ("every candidate, because round-1 scores were inflated" if len(keys) == len(cands) else "the top candidates")
            + " (round 2)."
        )
        if changed
        else "Nothing needs fixing, so round 2 fine-tunes the top candidates."
    )
    prev: dict[str, float] = {}
    for r in runs1:
        if np.isfinite(r.mean) and (
            r.candidate not in prev or (r.mean < prev[r.candidate] if metric in LOWER_IS_BETTER else r.mean > prev[r.candidate])
        ):
            prev[r.candidate] = r.mean
    board = Leaderboard(ctx, 2, [(k, cands[k].name, cands[k].family) for k in keys], metric, prev=prev, progress_span=(62, 82))
    ctx.model_update(stage="Model search, round 2")
    remaining = max(20.0, budget - (time.time() - ctx.state.get("train_started", time.time())))
    runs2 = supervised_round(
        task,
        cands,
        cfg2,
        2,
        time.perf_counter() + remaining * 0.85,
        board.progress,
        tracker(ctx),
        best_params(runs1, cands),
        keys,
        tune_top=1,
    )
    best = board.finish()
    ctx.mem["runs2"] = runs2
    ctx.state["round2"] = [r.summary() for r in runs2]
    lead1 = _leader(runs1)
    if best and lead1:
        ctx.say(
            f"Round 2: **{best['name']}** at {LABELS[metric]} **{best['display']}** (round 1 leader was {display(metric, lead1.mean)}"
            + (", inflated by the leaky column" if any(f.check == "leakage" and f.severity == "high" for f in applied) else "")
            + ")."
        )


def _forecast(ctx: StageContext, budget: int) -> None:
    task = ensure_fc_task(ctx)
    cfg = fc.FcConfig(**ctx.state["fc_cfg"])
    runs1: list[RunResult] = ctx.mem["runs1"]
    preds: dict[str, Any] = ctx.mem["preds"]
    with ctx.tool("critic.review(round=1)") as t:
        findings = fc.critic(task, runs1, preds, cfg)
        for f in findings:
            t.log(f"{f.severity:>6}  {f.title}")
    stop = next((f for f in findings if f.fix and f.fix.get("stop")), None)
    cfg2, applied = fc.apply_fixes(cfg, findings)
    leader = _leader(runs1)
    assert leader is not None
    _report(ctx, findings, leader, task.metric, applied)
    if stop:
        raise StopRunError(
            "no_signal",
            f"{stop.detail} Seasonal naive is the honest forecast here; more history or extra drivers would be needed to beat it.",
        )
    ctx.state["fc_cfg2"] = cfg2.to_dict()
    ctx.say(
        "Applying the fixes and re-running the backtest (round 2)."
        if applied
        else "Nothing needs fixing, so round 2 fine-tunes the best model."
    )
    prev = {r.candidate: r.mean for r in sorted(runs1, key=lambda r: r.selection_score)}
    cands = fc.forecast_candidates()
    board = Leaderboard(ctx, 2, [(c.key, c.name, c.family) for c in cands], task.metric, prev=prev, progress_span=(62, 82))
    remaining = max(15.0, budget - (time.time() - ctx.state.get("train_started", time.time())))
    runs2, preds2 = forecast_round(
        task,
        cfg2,
        ctx.mem["windows"],
        2,
        time.perf_counter() + remaining * 0.85,
        ctx.n_jobs,
        board.progress,
        tracker(ctx),
        fc_best_params(runs1),
    )
    board.finish()
    ctx.mem["runs2"], ctx.mem["preds"] = runs2, {**preds, **preds2}
    ctx.state["round2"] = [r.summary() for r in runs2]
