"""Stage 2: problem detection (heuristics + LLM, validated against the data), clarification and target confirmation."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ...db.repositories.activity import add_activity
from ...ml.detection import (
    UNUSABLE_TARGET,
    detect,
    name_leakage_suspects,
    positive_class,
    problem_for_target,
    series_column,
    target_candidates,
)
from ...ml.types import Detection
from ..context import StageContext, UserFacingError
from ..llm.base import FramingOut
from .common import problem_label


def _col(profile: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    return next((c for c in profile["columns"] if c["name"] == name), None)


def merge(det: Detection, ai: FramingOut | None, profile: dict[str, Any], df: pd.DataFrame) -> tuple[Detection, str | None, str | None]:
    """Accept the LLM framing only where it is consistent with the data. Returns (detection, title, question)."""
    if ai is None:
        return det, None, None
    names = {c["name"] for c in profile["columns"]}
    usable = {c["name"] for c in profile["columns"] if c["kind"] not in UNUSABLE_TARGET and not c.get("pii")}
    dates = {c["name"] for c in profile["columns"] if c["kind"] == "date"}
    numeric = {c["name"] for c in profile["columns"] if c["kind"] in ("number", "integer")}
    pt, tgt = ai.problem_type, ai.target
    valid = False
    if pt == "forecasting":
        valid = tgt in numeric and (ai.time_column in dates or det.time_column in dates)
    elif pt in ("binary", "multiclass", "regression"):
        valid = tgt in usable
    if valid and tgt:
        if pt == "forecasting":
            tc = ai.time_column if ai.time_column in dates else det.time_column
            sc = (
                ai.series_column if ai.series_column in names and ai.series_column not in (tgt, tc) else series_column(df, profile, tc, tgt)
            )
            det = Detection(
                "forecasting",
                tgt,
                tc,
                sc,
                None,
                None,
                ai.confidence,
                ai.signals or det.signals,
                ai.alternatives or det.alternatives,
                det.candidate_targets,
            )
        else:
            real = problem_for_target(profile, tgt)  # the data decides binary vs multiclass vs regression
            pos = (
                ai.positive_class
                if real == "binary" and ai.positive_class in set(df[tgt].dropna().astype(str))
                else (positive_class(df, tgt) if real == "binary" else None)
            )
            det = Detection(
                real,
                tgt,
                det.time_column,
                None,
                det.entity_column,
                pos,
                ai.confidence,
                ai.signals or det.signals,
                ai.alternatives or det.alternatives,
                det.candidate_targets,
            )
    elif pt is not None and not tgt:
        det.confidence = min(det.confidence, ai.confidence)
    suspects = {s["column"]: s for s in det.leakage_suspects}
    for s in ai.leakage_suspects:
        if s.column in names and s.column != det.target:
            suspects.setdefault(s.column, {"column": s.column, "reason": s.reason[:200]})
    det.leakage_suspects = list(suspects.values())[:5]
    if ai.horizon and ai.horizon > 0:
        det.horizon = int(ai.horizon)
    return det, (ai.title or None), ai.clarifying_question


def _apply_target(det: Detection, target: str, profile: dict[str, Any], df: pd.DataFrame) -> Detection:
    if det.problem_type == "forecasting":
        if _col(profile, target) and _col(profile, target)["kind"] in ("number", "integer"):  # type: ignore[index]
            det.target = target
            det.series_column = series_column(df, profile, det.time_column, target)
        return det
    det.target = target
    det.problem_type = problem_for_target(profile, target)
    det.positive_class = positive_class(df, target) if det.problem_type == "binary" else None
    det.leakage_suspects = name_leakage_suspects(profile, target)
    return det


def _describe_target(det: Detection, profile: dict[str, Any], df: pd.DataFrame) -> str:
    col = _col(profile, det.target) or {}
    if det.problem_type == "binary":
        share = (df[det.target].astype(str) == str(det.positive_class)).mean()
        return f"two values; `{det.positive_class}` is {share * 100:.1f}% of rows"
    if det.problem_type == "multiclass":
        return f"{col.get('unique')} categories"
    if "min" in col:
        return f"a number from {col['min']:,.4g} to {col['max']:,.4g}"
    return col.get("role", "")


def _title(goal: str, det: Detection, hint: str | None) -> str:
    if hint:
        return hint[:60]
    t = re.sub(r"[_\-.]+", " ", str(det.target or "Model")).strip().title()
    return f"{t} {'Forecast' if det.problem_type == 'forecasting' else 'Prediction' if det.problem_type != 'regression' else 'Estimator'}"


def run(ctx: StageContext) -> None:
    df, profile = ctx.df(), ctx.profile()
    ctx.thinking("Reading your goal and the columns…")
    with ctx.tool("problem_detector.analyze(goal, schema)") as t:
        det = detect(ctx.goal, profile, df)
        t.log(f"rules: {det.problem_type} on `{det.target or '?'}` (confidence {det.confidence:.2f})")
        ai = ctx.llm.frame_problem(ctx.goal, ctx.mem.get("summary", ""), det.to_dict())
        if ai is None:
            t.log("LLM: rule-based framing kept" + (" (MockLLM, no API key)" if ctx.llm.name == "mock" else " (LLM unavailable)"))
        else:
            t.log(f"LLM ({ctx.llm.model}): {ai.problem_type} on `{ai.target}` (confidence {ai.confidence:.2f})")
        det, title_hint, question = merge(det, ai, profile, df)
        t.log(f"decision: {det.problem_type} on `{det.target or '?'}`")
    numeric_only = det.problem_type == "forecasting"
    if det.target is None or det.confidence < 0.6:
        cands = [c for c in (det.candidate_targets or target_candidates(profile, numeric_only)) if c in df.columns][:5]
        if not cands:
            raise UserFacingError(
                "no_target", "This sheet has no column that could be predicted (every column is an identifier, free text or constant)."
            )
        r = ctx.ask(
            "options",
            question or "I'm not sure which column holds the outcome you want to predict. Which one is it?",
            options=[{"label": c, "value": c, "hint": (_col(profile, c) or {}).get("role")} for c in cands],
            default=cands[0],
        )
        if str(r["value"]) not in df.columns:
            raise UserFacingError("unknown_column", f"There is no column called '{r['value']}'.")
        det = _apply_target(det, str(r["value"]), profile, df)
        det.confidence = max(det.confidence, 0.8)
        det.signals = [f"You chose `{det.target}` as the outcome", *det.signals[:3]]

    title = _title(ctx.goal, det, title_hint)
    n_classes = int(df[det.target].nunique()) if det.problem_type == "multiclass" else None
    n_series = int(df[det.series_column].nunique()) if det.series_column else None
    plan = {**det.to_dict(), "title": title, "problem_label": problem_label(det.to_dict(), n_classes, n_series)}
    card_id = ctx.card("detection", _card(plan, df, profile))
    m = ctx.models.create(
        name=title,
        slug=re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60],
        problem_type=det.problem_type,
        problem_label=plan["problem_label"],
        objective=ctx.goal,
        status="draft",
        dataset_id=ctx.state["dataset_id"],
        session_id=ctx.session_id,
        target=det.target,
    )
    ctx.state["model_id"] = m.id
    ctx.set_status("running", model_id=m.id)
    ctx.model_update(stage="Problem detection", progress=6)
    add_activity(ctx.db, "agent", f"Started building **{title}**: {ctx.goal}", model_id=m.id, session_id=ctx.session_id)

    # ✋ confirm the target
    alts = [c for c in target_candidates(profile, numeric_only) if c != det.target][:3]
    r = ctx.ask(
        "options",
        f"I'll treat this as **{plan['problem_label'].lower()}** and predict `{det.target}` ({_describe_target(det, profile, df)}). Is that the right target?",
        options=[{"label": f"Yes, use {det.target}", "value": det.target}] + [{"label": f"Use {a} instead", "value": a} for a in alts],
        default=det.target,
    )
    chosen = str(r["value"])
    if chosen != det.target:
        if chosen not in df.columns:
            raise UserFacingError("unknown_column", f"There is no column called '{chosen}'.")
        det = _apply_target(det, chosen, profile, df)
        n_classes = int(df[det.target].nunique()) if det.problem_type == "multiclass" else None
        title = _title(ctx.goal, det, None)
        plan = {**plan, **det.to_dict(), "title": title, "problem_label": problem_label(det.to_dict(), n_classes, n_series)}
        ctx.card_update(card_id, _card(plan, df, profile))
        ctx.model_update(target=det.target, problem_type=det.problem_type, problem_label=plan["problem_label"], name=title)
        ctx.decision("Target", f"Target changed to `{det.target}`", "You chose a different outcome column.", "the detected target")
    ctx.state["plan"] = plan
    ctx.blueprint("Goal", ctx.goal)
    ctx.blueprint("Problem type", plan["problem_label"])
    ctx.blueprint("Target", det.target or "")
    if det.time_column:
        ctx.blueprint("Time column", det.time_column)
    ctx.decision(
        "Problem detection", f"{plan['problem_label']} on `{det.target}`", "; ".join(det.signals[:3]), "; ".join(det.alternatives[:2])
    )


def _card(plan: dict[str, Any], df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": plan["problem_type"],
        "problem_label": plan["problem_label"],
        "confidence": plan["confidence"],
        "target": plan["target"],
        "time_column": plan.get("time_column"),
        "series_column": plan.get("series_column"),
        "entity_column": plan.get("entity_column"),
        "positive_class": plan.get("positive_class"),
        "signals": plan["signals"],
        "alternatives": plan["alternatives"],
        "candidate_targets": plan.get("candidate_targets", []),
        "title": plan["title"],
    }
