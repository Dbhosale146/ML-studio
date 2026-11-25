"""Stage 4: cleaning plan (approved by the user) and feature engineering (LLM ideas validated by CV)."""

from __future__ import annotations

from typing import Any

from ...ml.features import FeatureBuilder, describe_derived, rule_based_derived, validate_derived
from ...ml.preprocessing import build_cleaning_plan, clean_rows, config_from_steps
from ...ml.training import evaluate_derived, prepare_task
from ...ml.types import CleaningStep, TrainConfig
from ..context import StageContext
from .common import ensure_candidates, ensure_fc_task, strategy_of


def run(ctx: StageContext) -> None:
    plan = ctx.state["plan"]
    if plan["problem_type"] == "forecasting":
        _forecast(ctx)
        return
    df, profile = ctx.df(), ctx.profile()
    steps = build_cleaning_plan(df, profile, plan["target"], [])
    items = [s.to_dict() for s in steps]
    card_id = ctx.card("cleaning", {"items": items, "approved": False})
    optional = [s.id for s in steps if s.optional]
    # ✋ approve the cleaning plan
    r = ctx.ask(
        "approval",
        "Here is my cleaning plan with exact counts. Untick any optional fix you want me to skip, then approve.",
        options=[{"label": "Approve plan", "value": optional}],
        items=items,
        default=optional,
    )
    chosen = r["value"] if isinstance(r["value"], list) else optional
    for s in steps:
        s.enabled = (not s.optional) or s.id in chosen
    skipped = [s for s in steps if s.optional and not s.enabled]
    for s in skipped:
        ctx.decision("Cleaning", f"Skipped: {s.title}", "You chose to keep the data as is for this step.")
    ctx.card_update(card_id, {"items": [s.to_dict() for s in steps], "approved": True})
    ctx.state["cleaning"] = [s.to_dict() for s in steps]
    cfg = config_from_steps(steps, TrainConfig())
    dedupe = any(s.id == "dedupe_rows" and s.enabled for s in steps)
    with ctx.tool("data_cleaner.apply(plan)") as t:
        clean, stats = clean_rows(df, plan["target"], dedupe=dedupe)
        t.log(f"rows: {stats['rows_in']:,} → {stats['rows_out']:,}")
        if stats["dropped_duplicates"]:
            t.log(f"removed {stats['dropped_duplicates']:,} duplicate rows")
        if stats["dropped_missing_target"]:
            t.log(f"dropped {stats['dropped_missing_target']:,} rows without {plan['target']}")
        for s in steps:
            if s.enabled and s.id not in ("dedupe_rows", "drop_missing_target"):
                t.log(f"{s.title} (learned on training folds only)")
    ctx.mem["clean"] = clean
    ctx.state["dedupe"] = dedupe
    ctx.decision("Cleaning", f"Applied {sum(s.enabled for s in steps)} fixes", "; ".join(s.title for s in steps if s.enabled)[:400])

    task = prepare_task(
        clean,
        profile,
        plan["problem_type"],
        plan["target"],
        strategy_of(ctx),
        [],
        plan.get("time_column"),
        plan.get("positive_class"),
        ctx.settings.random_state,
    )
    ctx.mem["task"] = task
    if task.dropped_classes:
        ctx.say(
            f"I left out {len(task.dropped_classes)} rare class(es) with too few rows to learn from: {', '.join(task.dropped_classes[:5])}."
        )
    cands = ensure_candidates(ctx)
    with ctx.tool("feature_engineer.synthesize()") as t:
        numeric = [c for c in task.columns if task.kinds[c] in ("number", "integer")]
        ideas = ctx.llm.suggest_features(ctx.goal, plan["target"], numeric, ctx.mem.get("summary", ""))
        source = {i.get("name"): i.get("source", "llm") for i in ideas}
        ideas = ideas + rule_based_derived(task.columns, task.kinds, profile["columns"], plan["target"])
        specs = validate_derived(ideas, task.columns, task.kinds, plan["target"])
        for sp in specs:
            sp["source"] = source.get(sp["name"], "rules")
        t.log(f"{len(ideas)} ideas ({sum(1 for sp in specs if sp['source'] == 'llm')} from the LLM), {len(specs)} valid")
        quick = cands.get("hgb") or cands.get("lgbm") or next(c for c in cands.values() if not c.baseline)
        accepted = evaluate_derived(task, quick, cfg, specs) if specs else []
        ok = {sp["name"]: g for sp, g in accepted}
        for sp in specs:
            t.log(
                f"{'kept' if sp['name'] in ok else 'rejected'} {sp['name']} = {describe_derived(sp)}"
                + (f" (+{ok[sp['name']]:.4f} CV)" if sp["name"] in ok else " (no CV gain)")
            )
        cfg.derived = [sp for sp in specs if sp["name"] in ok]
        fb = FeatureBuilder(
            task.columns, task.kinds, cfg.derived, cfg.cap_outliers, cfg.missing_flags, cfg.normalise_labels, cfg.group_rare
        ).fit(task.X_search.head(2000))
        t.log(f"{len(fb.feature_names_out_)} model features from {len(task.columns)} input columns")
    feat_items: list[dict[str, Any]] = [{"name": c, "kind": "raw", "desc": task.kinds[c]} for c in task.columns]
    for c in fb.date_cols_:
        feat_items.append({"name": f"{c} → year, month, weekday, cyclical", "kind": "date", "desc": "date parts with sin/cos seasonality"})
    for c in fb.miss_cols_:
        feat_items.append({"name": f"{c}__missing", "kind": "missing", "desc": "flag: value was empty"})
    for sp in specs:
        feat_items.append(
            {
                "name": sp["name"],
                "kind": "derived",
                "desc": describe_derived(sp),
                "why": sp.get("why", ""),
                "source": sp["source"],
                "accepted": sp["name"] in ok,
                "gain": ok.get(sp["name"]),
            }
        )
    features = {
        "items": feat_items,
        "total": len(fb.feature_names_out_),
        "inputs": len(task.columns),
        "accepted": len(ok),
        "proposed": len(specs),
    }
    ctx.card("features", features)
    ctx.state["features"] = features
    ctx.state["cfg"] = cfg.to_dict()
    ctx.blueprint(
        "Features",
        f"{len(fb.feature_names_out_)} model features from {len(task.columns)} columns" + (f" · {len(ok)} derived" if ok else ""),
    )
    if specs:
        ctx.decision(
            "Features",
            f"Kept {len(ok)} of {len(specs)} derived features",
            "A derived feature is only kept when it improves the cross-validated score." + (f" Kept: {', '.join(ok)}." if ok else ""),
        )


def _forecast(ctx: StageContext) -> None:
    from ...ml import forecasting as fc

    task = ensure_fc_task(ctx)
    plan = ctx.state["plan"]
    steps = [CleaningStep(f"fc_{i}", n["issue"], n["action"], int(n["count"]), optional=False) for i, n in enumerate(task.notes)]
    steps.append(
        CleaningStep(
            "fc_regular",
            f"Build {len(task.series)} regular {task.unit}ly timeline{'s' if len(task.series) > 1 else ''}",
            f"{len(task.panel):,} points after cleaning",
            len(task.panel),
            optional=False,
        )
    )
    items = [s.to_dict() for s in steps]
    card_id = ctx.card("cleaning", {"items": items, "approved": False})
    ctx.ask(
        "approval",
        "This is how I'll prepare the time series. Approve to continue.",
        options=[{"label": "Approve", "value": []}],
        items=items,
        default=[],
    )
    ctx.card_update(card_id, {"items": items, "approved": True})
    ctx.state["cleaning"] = items
    cfg = fc.default_config(task)
    with ctx.tool("feature_engineer.time_series()") as t:
        t.log(f"lags {cfg.lags} (all ≥ horizon {task.horizon}, so nothing leaks)")
        t.log(f"rolling means over {cfg.windows} {task.unit}s ending {task.horizon} {task.unit}s back")
        t.log("calendar: seasonal sin/cos encodings, trend index, series id")
    items2: list[dict[str, Any]] = [{"name": f"lag_{k}", "kind": "lag", "desc": f"value {k} {task.unit}s earlier"} for k in cfg.lags]
    items2 += [
        {"name": f"roll_mean_{w}", "kind": "rolling", "desc": f"{w}-{task.unit} average ending {task.horizon} back"} for w in cfg.windows
    ]
    items2 += [
        {"name": "seasonal encodings", "kind": "calendar", "desc": "week/month of year as sin/cos"},
        {"name": "series_code", "kind": "raw", "desc": "which series"},
    ]
    ctx.state["features"] = {"items": items2, "total": len(items2) + 2, "inputs": 1, "accepted": 0, "proposed": 0}
    ctx.card("features", ctx.state["features"])
    ctx.state["fc_cfg"] = cfg.to_dict()
    ctx.blueprint("Features", f"{len(cfg.lags)} lags ≥ {task.horizon}, {len(cfg.windows)} rolling windows, calendar")
    ctx.decision(
        "Features",
        f"Lags {cfg.lags}",
        f"Every lag is at least the horizon ({task.horizon}), so one direct model forecasts all steps without future data.",
    )
    plan["n_series"] = len(task.series)
