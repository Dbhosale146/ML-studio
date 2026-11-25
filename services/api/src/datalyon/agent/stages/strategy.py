"""Stage 3: metric and validation from rules (horizon and frequency asked for forecasting); the LLM explains."""

from __future__ import annotations

from ...ml.metrics import LABELS
from ...ml.strategy import FREQ_INFO, choose_strategy, horizon_from_goal, infer_frequency
from ..context import StageContext, UserFacingError


def run(ctx: StageContext) -> None:
    plan = ctx.state["plan"]
    df = ctx.df()
    if plan["problem_type"] == "forecasting":
        freq = plan.get("freq") or infer_frequency(df[plan["time_column"]])
        if freq is None:
            r = ctx.ask(
                "options",
                f"I couldn't infer how often `{plan['time_column']}` is recorded. What is the frequency of the data?",
                options=[{"label": v["label"], "value": k} for k, v in FREQ_INFO.items()],
                default="D",
            )
            freq = str(r["value"])
        info = FREQ_INFO[freq]
        per_series = df.groupby(plan["series_column"]).size().min() if plan.get("series_column") else df[plan["time_column"]].nunique()
        max_h = max(1, int(per_series) // 3)
        suggested = plan.get("horizon") or horizon_from_goal(ctx.goal, freq) or int(info["default_h"])
        opts = sorted({h for h in [suggested, *info["options"]] if 1 <= h <= max_h})
        if not opts:
            raise UserFacingError(
                "history_too_short", f"There are only {per_series} {info['unit']}s of history per series, which is too short to forecast."
            )
        # ✋ horizon
        r = ctx.ask(
            "options",
            f"Your data is **{info['label'].lower()}**. How far ahead should I forecast?",
            options=[
                {"label": f"{h} {info['unit']}{'s' if h > 1 else ''}", "value": h, "hint": "from your goal" if h == suggested else None}
                for h in opts
            ],
            default=suggested if suggested in opts else opts[-1],
        )
        horizon = max(1, min(int(r["value"]), max_h))
        plan.update(freq=freq, horizon=horizon)
        ctx.blueprint("Horizon", f"{horizon} {info['unit']}{'s' if horizon > 1 else ''} ({info['label'].lower()})")
        if plan.get("series_column"):
            ctx.blueprint("Series", f"{df[plan['series_column']].nunique()} × `{plan['series_column']}` (one global model)")
    strat = choose_strategy(
        plan["problem_type"],
        df,
        plan["target"],
        plan.get("time_column") if plan["problem_type"] != "forecasting" else plan["time_column"],
        ctx.settings.train_time_budget_s,
        plan.get("positive_class"),
        plan.get("horizon"),
        plan.get("freq"),
    )
    ctx.thinking("Choosing the metric and validation scheme…")
    explanation = ctx.llm.explain_strategy(
        {
            "problem": plan["problem_label"],
            "metric": strat.metric_label,
            "validation": strat.validation,
            "reasons": strat.reasons,
            "extras": strat.extras,
        }
    )
    ctx.card(
        "strategy",
        {
            "metric": strat.metric,
            "metric_label": strat.metric_label,
            "better": "higher" if strat.greater_is_better else "lower",
            "secondary": [LABELS.get(m, m) for m in strat.secondary],
            "validation": strat.validation,
            "explanation": explanation,
            "reasons": strat.reasons,
            "extras": [list(x) for x in strat.extras],
            "time_budget_s": strat.time_budget_s,
            "class_weighting": strat.class_weighting,
        },
    )
    ctx.state["strategy"] = strat.to_dict()
    ctx.state["plan"] = plan
    ctx.blueprint("Metric", f"{strat.metric_label} ({'higher' if strat.greater_is_better else 'lower'} is better)")
    ctx.blueprint("Validation", strat.validation)
    ctx.decision(
        "Strategy",
        f"Optimise {strat.metric_label}; {strat.validation}",
        strat.reasons[0] if strat.reasons else "",
        ", ".join(LABELS.get(m, m) for m in strat.secondary),
    )
