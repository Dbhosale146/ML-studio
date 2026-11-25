"""Stage 8: refit on all rows, save a versioned artifact + model card + input schema, mark the model ready."""

from __future__ import annotations

import time
from typing import Any

from ...db.repositories.activity import add_activity
from ...ml.pipeline import build_tabular_artifact, make_card
from ...ml.registry import save_version
from ..context import StageContext
from .common import ResumeFromError, ensure_candidates, ensure_fc_task, ensure_task, strategy_of, tracker


def run(ctx: StageContext) -> None:
    plan = ctx.state["plan"]
    forecasting = plan["problem_type"] == "forecasting"
    if (not forecasting and "final" not in ctx.mem) or (forecasting and "fc_artifact" not in ctx.mem):
        raise ResumeFromError("train")
    mid = ctx.state["model_id"]
    ds = ctx.datasets.get(ctx.state["dataset_id"])
    final_state = ctx.state["final"]
    t0 = time.time()
    with ctx.tool("model_registry.publish()") as t:
        if forecasting:
            task = ensure_fc_task(ctx)
            art = ctx.mem["fc_artifact"]
            columns: list[str] = [plan["time_column"], plan["target"]] + ([plan["series_column"]] if plan.get("series_column") else [])
            schema_doc: dict[str, Any] = {
                "fields": [],
                "examples": [],
                "forecast": {"series": list(art["forecast"]["chart"]["series"].keys()), "horizon": task.horizon, "unit": task.unit},
            }
            n_rows = len(task.panel)
        else:
            task_ = ensure_task(ctx)
            cands = ensure_candidates(ctx)
            final = ctx.mem["final"]
            columns = [c for c in task_.columns if c not in final.config.exclude]
            labels = ctx.llm.field_labels(columns, ctx.goal, plan["target"])
            art, schema, secs = build_tabular_artifact(task_, final, cands, ctx.mem["clean"], labels)
            t.log(f"refitted {final.candidate} on all {len(task_.y_train) + len(task_.y_hold):,} rows in {secs:.1f}s")
            schema_doc = {"fields": schema, "examples": art["examples"]}
            n_rows = len(task_.y_train) + len(task_.y_hold)
        version = ctx.models.next_version(mid)
        excluded = [{"column": c, "reason": "leakage (critic)"} for c in (ctx.state.get("cfg2") or {}).get("exclude", [])]
        card = make_card(
            model_name=plan["title"],
            problem_type=plan["problem_type"],
            goal=ctx.goal,
            target=plan["target"],
            algorithm=final_state["algorithm"],
            params=final_state.get("params", {}),
            config=final_state.get("config", ctx.state.get("fc_cfg2") or {}),
            strategy=strategy_of(ctx),
            metrics=final_state["metrics"],
            data_hash=ds.data_hash or "",
            dataset={"id": ds.id, "file_name": ds.file_name, "sheet": ds.sheet, "rows": ds.rows, "cols": ds.cols, "training_rows": n_rows},
            columns=columns,
            excluded=excluded,
            training_seconds=time.time() - ctx.state.get("train_started", t0),
            seed=ctx.settings.random_state,
            extra={"version": version, "llm": {"provider": ctx.llm.name, "model": ctx.llm.model}, "ui": _ui(ctx, ds)},
        )
        path = save_version(ctx.settings.artifacts_dir, mid, version, art, card, schema_doc)
        t.log(f"saved version v{version}: pipeline, model card and input schema")
        ctx.models.add_version(mid, version, str(path), card, schema_doc, ds.data_hash or "")
    tracker(ctx).log_run(
        f"{plan['title']} v{version}",
        {
            "algorithm": final_state["algorithm"],
            "data_hash": ds.data_hash,
            **{f"param_{k}": v for k, v in final_state.get("params", {}).items()},
        },
        {m["key"]: m["value"] for m in final_state["metrics"] if isinstance(m.get("value"), (int, float))},
        {"model_id": mid, "version": str(version)},
    )
    p = final_state["primary"]
    ctx.model_update(
        status="ready",
        progress=100,
        stage="Ready to test",
        latest_version=version,
        algorithm=final_state["algorithm"],
        primary_metric={"name": p["name"], "display": p["display"], "value": p.get("value")},
        baseline_metric={"name": p["name"], "display": p["baseline"], "value": None},
    )
    add_activity(
        ctx.db,
        "ready",
        f"**{plan['title']}** v{version} is ready to test ({p['name']} {p['display']})",
        model_id=mid,
        session_id=ctx.session_id,
    )
    ctx.card(
        "published",
        {
            "model_id": mid,
            "version": version,
            "name": plan["title"],
            "algorithm": final_state["algorithm"],
            "primary": p,
            "test_url": f"/models/{mid}?tab=test",
            "artifact": f"models/{mid}/v{version}",
        },
    )
    ctx.blueprint("Published", f"v{version} · ready to test")
    ctx.say(
        f"Done! **{plan['title']}** v{version} is saved and ready. Open **Test this model** to "
        + (
            "pick a series and a horizon, see the forecast with its intervals and download it."
            if forecasting
            else "try it with your own inputs or score a whole Excel file."
        )
    )


def _ui(ctx: StageContext, ds: Any) -> dict[str, Any]:
    """Everything the model detail page shows, captured at publish time."""
    s = ctx.state
    profile = ctx.profile()
    return {
        "plan": {
            k: s["plan"].get(k)
            for k in (
                "problem_type",
                "problem_label",
                "target",
                "positive_class",
                "time_column",
                "series_column",
                "entity_column",
                "signals",
                "alternatives",
                "confidence",
                "horizon",
                "freq",
            )
        },
        "dataset": {
            "file_name": ds.file_name,
            "sheet": ds.sheet,
            "rows": profile.get("rows"),
            "cols": profile.get("cols"),
            "quality": profile.get("quality"),
            "quality_reasons": profile.get("quality_reasons"),
            "period": profile.get("period"),
            "columns": [
                {k: c.get(k) for k in ("name", "kind", "role", "missing_pct", "unique", "pii")} for c in profile.get("columns", [])
            ],
        },
        "strategy": s.get("strategy"),
        "cleaning": s.get("cleaning", []),
        "features": s.get("features", {}),
        "round1": s.get("round1", []),
        "round2": s.get("round2", []),
        "critic": s.get("critic"),
        "final": s.get("final"),
        "decisions": s.get("decisions", []),
        "blueprint": s.get("blueprint", {}),
    }
