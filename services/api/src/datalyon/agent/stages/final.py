"""Stage 7: final evaluation on the holdout (or the backtest for forecasting), explanation and the user's acceptance."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...ml import forecasting as fc
from ...ml.evaluation import FinalResult, choose_final
from ...ml.metrics import LABELS, display
from ...ml.pipeline import build_forecast_artifact
from ...ml.types import RunResult
from ..context import StageContext
from .common import ResumeFromError, ensure_candidates, ensure_fc_task, ensure_task, train_config


def _final_payload(ctx: StageContext, final: FinalResult, metric: str) -> dict[str, Any]:
    task = ensure_task(ctx)
    base = final.baseline.get(metric)
    facts = {
        "algorithm": final.candidate,
        "metric": LABELS[metric],
        "score": display(metric, final.holdout.get(metric)),
        "baseline": display(metric, base),
        "runner_up": final.runner_up["candidate"] if final.runner_up else None,
        "runner_up_score": display(metric, final.runner_up["holdout"]) if final.runner_up else None,
        "top_features": [c for c, _ in final.importance[:5]],
        "holdout_metrics": {m["name"]: m["display"] for m in final.metrics},
        "cv": f"{display(metric, final.cv_mean)} ± {display(metric, final.cv_std)}",
        "target": task.target,
        "problem": ctx.state["plan"]["problem_label"],
    }
    with ctx.tool("explainer.write(model_card)"):
        expl = ctx.llm.explain_model(facts)
    return {
        "algorithm": final.candidate,
        "problem_type": task.problem_type,
        "primary": {
            "key": metric,
            "name": LABELS[metric],
            "display": display(metric, final.holdout.get(metric)),
            "value": final.holdout.get(metric),
            "baseline": display(metric, base),
        },
        "metrics": final.metrics,
        "charts": final.charts,
        "importance": [[c, float(v)] for c, v in final.importance[:15]],
        "explanation": expl.explanation,
        "why_model": expl.why_model,
        "ranking": final.ranking,
        "threshold": final.threshold if task.problem_type == "binary" else None,
        "classes": task.classes,
        "cv": {
            "mean": final.cv_mean,
            "std": final.cv_std,
            "display": f"{display(metric, final.cv_mean)} ± {display(metric, final.cv_std)}",
        },
        "holdout_rows": len(task.y_hold),
        "notes": final.notes,
        "runner_up": final.runner_up,
        "params": final.params,
        "config": final.config.to_dict(),
        "candidate_key": final.candidate_key,
    }


def run(ctx: StageContext) -> None:
    if "runs2" not in ctx.mem:
        raise ResumeFromError("train")
    ctx.model_update(stage="Final evaluation", progress=85)
    if ctx.state["plan"]["problem_type"] == "forecasting":
        _forecast(ctx)
        return
    task, cands = ensure_task(ctx), ensure_candidates(ctx)
    cfg, cfg2 = train_config(ctx, "cfg"), train_config(ctx, "cfg2")
    runs1: list[RunResult] = ctx.mem["runs1"]
    runs2: list[RunResult] = ctx.mem["runs2"]
    pool = runs2 if set(cfg2.exclude) != set(cfg.exclude) else runs1 + runs2  # round-1 scores with a leaky column are not comparable
    metric = task.strategy.metric
    with ctx.tool("evaluator.holdout(top 3)") as t:
        final = choose_final(task, pool, cands, cfg2)
        for r in final.ranking:
            t.log(f"{r['candidate']:<28} holdout {display(metric, r['holdout'])}" + (" ← selected" if r["selected"] else ""))
    payload = _final_payload(ctx, final, metric)
    card_id = ctx.card("final", payload)
    ctx.blueprint("Best model", f"{final.candidate} · {LABELS[metric]} {payload['primary']['display']}")
    ctx.say(
        f"**{final.candidate}** wins with {LABELS[metric]} **{payload['primary']['display']}** on {len(task.y_hold):,} held-out rows, "
        f"against {payload['primary']['baseline']} for the baseline. {payload['explanation']}"
    )
    while True:
        options = [{"label": "Accept and publish", "value": "accept"}]
        if final.runner_up:
            options.append({"label": f"Use {final.runner_up['candidate']} instead", "value": "runner_up"})
        r = ctx.ask("options", "Do you want to publish this model for testing?", options=options, default="accept")
        if r["value"] != "runner_up" or not final.runner_up:
            break
        name = final.runner_up["candidate"]
        with ctx.tool(f"evaluator.holdout({name})"):
            final = choose_final(task, [x for x in pool if x.candidate == name], cands, cfg2)
            final.runner_up = None
        payload = _final_payload(ctx, final, metric)
        ctx.card_update(card_id, payload)
        ctx.decision("Final model", f"Switched to {name}", "You preferred the runner-up.")
        ctx.say(f"Switched to **{name}**: {LABELS[metric]} {payload['primary']['display']} on the holdout.")
    ctx.mem["final"] = final
    ctx.state["final"] = {k: v for k, v in payload.items()}
    ctx.decision(
        "Final model",
        f"{final.candidate} ({LABELS[metric]} {payload['primary']['display']})",
        payload["why_model"],
        ", ".join(r["candidate"] for r in final.ranking if not r["selected"]),
    )


def _forecast(ctx: StageContext) -> None:
    task = ensure_fc_task(ctx)
    runs: list[RunResult] = ctx.mem["runs1"] + ctx.mem["runs2"]
    with ctx.tool("evaluator.backtest_and_forecast()") as t:
        artifact, metrics, algorithm = build_forecast_artifact(task, runs, ctx.mem["preds"], ctx.n_jobs)
        t.log(f"refitted {algorithm} on the full history and forecast {task.horizon} {task.unit}s ahead for {len(task.series)} series")
    f = artifact["forecast"]
    importance: list[list[Any]] = []
    model = artifact["model"]
    if model is not None and hasattr(model, "feature_importances_"):
        cfg = fc.FcConfig(**f["config"])
        cols = fc.feature_columns(fc.build_features(task.panel.head(200), task, cfg, {s: i for i, s in enumerate(task.series)}))
        imp = np.asarray(model.feature_importances_, dtype=float)
        if imp.sum() > 0:
            ranked = sorted(zip(cols, (float(v / imp.sum()) for v in imp), strict=True), key=lambda x: -x[1])[:12]
            importance = [[c, v] for c, v in ranked]
    primary = metrics[0]
    facts = {
        "algorithm": algorithm,
        "metric": primary["name"],
        "score": primary["display"],
        "baseline": primary["baseline"],
        "runner_up": None,
        "top_features": [c for c, _ in importance[:5]],
        "holdout_metrics": {m["name"]: m["display"] for m in metrics},
        "series": len(task.series),
        "horizon": f"{task.horizon} {task.unit}s",
    }
    with ctx.tool("explainer.write(model_card)"):
        expl = ctx.llm.explain_model(facts)
    payload = {
        "algorithm": algorithm,
        "problem_type": "forecasting",
        "primary": {
            "key": primary["key"],
            "name": primary["name"],
            "display": primary["display"],
            "value": primary["value"],
            "baseline": primary["baseline"],
        },
        "metrics": metrics,
        "charts": {"forecast": f["chart"], "backtest": f["backtest"], "metric": f["metric"]},
        "importance": importance,
        "explanation": expl.explanation,
        "why_model": expl.why_model,
        "ranking": [],
        "horizon": task.horizon,
        "unit": task.unit,
        "series": task.series,
        "notes": [],
    }
    ctx.card("final", payload)
    ctx.blueprint("Best model", f"{algorithm} · {primary['name']} {primary['display']}")
    ctx.say(
        f"**{algorithm}** wins with backtest {primary['name']} **{primary['display']}** vs {primary['baseline']} for seasonal naive. {expl.explanation}"
    )
    ctx.ask(
        "options",
        "Do you want to publish this forecasting model for testing?",
        options=[{"label": "Accept and publish", "value": "accept"}],
        default="accept",
    )
    ctx.mem["fc_artifact"] = artifact
    ctx.state["final"] = payload
    ctx.decision("Final model", f"{algorithm} ({primary['name']} {primary['display']})", expl.why_model)
