"""Heuristic problem and target detection from the goal wording and the data profile.

Used on its own with the MockLLM, and as the validator / fallback for the LLM's proposal.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .types import Detection, ProblemType

STOP = set(
    [
        "predict",
        "prediction",
        "which",
        "what",
        "will",
        "would",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "by",
        "is",
        "are",
        "be",
        "next",
        "who",
        "whether",
        "if",
        "from",
        "my",
        "our",
        "and",
        "or",
        "with",
        "per",
        "using",
        "based",
        "each",
        "want",
        "need",
        "like",
        "know",
        "how",
        "much",
        "many",
        "does",
        "do",
        "can",
        "could",
        "should",
        "find",
        "estimate",
        "forecast",
        "build",
        "model",
        "me",
        "us",
        "it",
        "this",
        "that",
        "these",
        "those",
        "their",
        "there",
        "likely",
    ]
)
SYNONYMS = [
    {
        "churn",
        "churned",
        "attrition",
        "cancel",
        "cancelled",
        "canceled",
        "cancellation",
        "left",
        "exit",
        "exited",
        "retain",
        "retention",
        "leave",
    },
    {"price", "cost", "value", "amount", "valuation", "worth"},
    {"sale", "sales", "revenue", "unit", "units", "quantity", "qty", "demand", "order", "orders", "volume", "sold"},
    {"fraud", "fraudulent", "anomaly", "suspicious", "scam"},
    {"default", "defaulted", "delinquent", "delinquency", "repay", "repaid"},
    {"convert", "converted", "conversion", "purchase", "purchased", "bought", "signup", "subscribe"},
    {"category", "categories", "class", "type", "segment", "label"},
]
POSITIVE = {
    "1",
    "1.0",
    "yes",
    "true",
    "y",
    "t",
    "churned",
    "churn",
    "fraud",
    "default",
    "defaulted",
    "positive",
    "won",
    "converted",
    "bad",
    "failed",
    "left",
    "exited",
}
TARGET_HINTS = re.compile(
    r"churn|target|label|class|category|outcome|default|fraud|price|sales|revenue|amount|demand|quantity|qty|units|status|convert|"
    r"response|score|risk|result|win|won|cancel|attrition|purchase|value|cost|profit|rating|approved|failure|defect|spend|income|salary|^y$",
    re.I,
)
FORECAST_INTENT = re.compile(
    r"\b(forecast\w*|predict (the )?(future|next)|next \d+ (hour|day|week|month|quarter|year)s?|next (hour|day|week|month|quarter|year)|"
    r"future|projection|upcoming|coming (weeks|months|days|quarters?))\b",
    re.I,
)
UNUSABLE_TARGET = ("id", "text", "constant", "empty", "date")
# words that mark a column as recorded after the outcome (e.g. cancellation_reason for churn)
AFTER_THE_FACT = {
    _w
    for _w in [
        "reason",
        "status",
        "outcome",
        "result",
        "final",
        "actual",
        "closed",
        "close",
        "resolution",
        "resolved",
        "after",
        "post",
        "note",
        "notes",
        "comment",
        "code",
        "flag",
        "confirmed",
        "confirm",
    ]
}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return {_stem(t) for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t and t not in STOP}


def expand(toks: set[str]) -> set[str]:
    out = set(toks)
    for group in SYNONYMS:
        stems = {_stem(g) for g in group}
        if toks & stems:
            out |= stems
    return out


def _col(profile: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    return next((c for c in profile["columns"] if c["name"] == name), None)


def target_candidates(profile: dict[str, Any], numeric_only: bool = False) -> list[str]:
    cols = profile["columns"]
    scored: list[tuple[float, str]] = []
    for i, c in enumerate(cols):
        if c["kind"] in UNUSABLE_TARGET:
            continue
        if numeric_only and c["kind"] not in ("number", "integer"):
            continue
        score = 0.0
        if TARGET_HINTS.search(c["name"]):
            score += 3
        if c["kind"] == "boolean" or (c["kind"] in ("category", "integer") and c["unique"] == 2):
            score += 2
        if c["kind"] in ("number", "integer") and c["unique"] > 10:
            score += 1
        if c["kind"] == "category" and 2 < c["unique"] <= 20:
            score += 1
        if i == len(cols) - 1:
            score += 1.5
        score -= c["missing_pct"] / 20
        if c.get("pii"):
            score -= 5
        scored.append((score, c["name"]))
    scored.sort(key=lambda t: -t[0])
    return [n for _, n in scored]


def problem_for_target(profile: dict[str, Any], target: str) -> ProblemType:
    c = _col(profile, target)
    if c is None:
        return "regression"
    if c["kind"] == "boolean" or c["unique"] == 2:
        return "binary"
    if c["kind"] == "category":
        return "multiclass"
    if (
        c["kind"] == "integer"
        and c["unique"] <= 10
        and re.search(r"class|grade|level|rating|tier|category|segment|stage|type", target, re.I)
    ):
        return "multiclass"
    return "regression"


def match_column(goal: str, profile: dict[str, Any], allowed: set[str]) -> tuple[str | None, float]:
    gt = expand(tokens(goal))
    goal_norm = re.sub(r"[^a-z0-9]", "", goal.lower())
    best, best_score = None, 0.0
    for c in profile["columns"]:
        if c["name"] not in allowed:
            continue
        name_norm = re.sub(r"[^a-z0-9]", "", c["name"].lower())
        score = float(len(tokens(c["name"]) & gt))
        if len(name_norm) > 3 and name_norm in goal_norm:
            score += 3
        if score > best_score:
            best, best_score = c["name"], score
    return best, best_score


def main_date_column(profile: dict[str, Any]) -> str | None:
    dates = [c for c in profile["columns"] if c["kind"] == "date"]
    if not dates:
        return None
    named = [c for c in dates if re.search(r"date|time|week|month|period|day|year", c["name"], re.I)]
    return str(max(named or dates, key=lambda c: c["unique"])["name"])


def series_column(df: pd.DataFrame, profile: dict[str, Any], time_col: str | None, target: str | None) -> str | None:
    """A column that splits the data into several time series (e.g. store_id)."""
    if time_col is None or not df[time_col].duplicated().any():
        return None
    best: tuple[str, int] | None = None
    for c in profile["columns"]:
        if c["name"] in (time_col, target) or c["kind"] not in ("category", "integer", "boolean", "id"):
            continue
        if not (2 <= c["unique"] <= 1000):
            continue
        if df.duplicated([time_col, c["name"]]).mean() <= 0.01 and (best is None or c["unique"] < best[1]):
            best = (c["name"], c["unique"])
    return best[0] if best else None


def positive_class(df: pd.DataFrame, target: str) -> str | None:
    vc = df[target].dropna().astype(str).value_counts()
    if len(vc) != 2:
        return None
    for v in vc.index:
        if str(v).strip().lower() in POSITIVE:
            return str(v)
    return str(vc.index[-1])  # minority class


def entity_column(profile: dict[str, Any]) -> str | None:
    ids = [c["name"] for c in profile["columns"] if c["kind"] == "id"]
    return ids[0] if ids else None


def name_leakage_suspects(profile: dict[str, Any], target: str | None) -> list[dict[str, str]]:
    """Columns whose name relates to the target and adds a qualifier (reason, date, status...): often filled after the outcome."""
    if not target:
        return []
    tt = expand(tokens(target))
    out = []
    for c in profile["columns"]:
        if c["name"] == target or c["kind"] in ("id", "constant", "empty"):
            continue
        ct = tokens(c["name"])
        if ct & tt and (ct - tt) & AFTER_THE_FACT:
            out.append(
                {
                    "column": c["name"],
                    "reason": f"its name relates to the target `{target}`, so it may only be filled in after the outcome is known",
                }
            )
    return out[:3]


def detect(goal: str, profile: dict[str, Any], df: pd.DataFrame) -> Detection:
    usable = {c["name"] for c in profile["columns"] if c["kind"] not in UNUSABLE_TARGET and not c.get("pii")}
    time_col = main_date_column(profile)
    alternatives: list[str] = []

    if FORECAST_INTENT.search(goal) and time_col:
        numeric = {c["name"] for c in profile["columns"] if c["kind"] in ("number", "integer")}
        t, s = match_column(goal, profile, numeric)
        if s < 1:
            t = next(iter(target_candidates(profile, numeric_only=True)), None)
        if t:
            sc = series_column(df, profile, time_col, t)
            fc_signals = [
                "Your goal is about future values over time",
                f"`{time_col}` is a date column that orders the history",
                f"`{t}` is numeric",
            ]
            if sc:
                fc_signals.append(f"`{sc}` splits the data into {df[sc].nunique()} separate series")
            return Detection(
                problem_type="forecasting",
                target=t,
                time_column=time_col,
                series_column=sc,
                entity_column=None,
                confidence=0.88 if s >= 1 else 0.6,
                signals=fc_signals,
                alternatives=["Regression on each row: rejected because it ignores time order and seasonality."],
                candidate_targets=target_candidates(profile, numeric_only=True)[:5],
                leakage_suspects=[],
            )

    target, score = match_column(goal, profile, usable)
    signals: list[str] = []
    if target and score >= 1:
        signals.append(f"Column `{target}` matches the wording of your goal")
    else:
        target = None
    cands = target_candidates(profile)
    if target is None and cands:
        top = cands[0]
        if TARGET_HINTS.search(top) and (len(cands) == 1 or (TARGET_HINTS.search(top) and not TARGET_HINTS.search(cands[1]))):
            target = top
            signals.append(f"`{top}` is the column that looks most like an outcome")
            score = 0.5
    if target is None:
        return Detection(
            problem_type="binary",
            target=None,
            time_column=time_col,
            entity_column=entity_column(profile),
            confidence=0.3,
            signals=["I could not tell from your goal which column is the outcome"],
            candidate_targets=cands[:5],
            clarification="Which column holds the outcome you want to predict?",
        )

    ptype = problem_for_target(profile, target)
    col = _col(profile, target)
    assert col is not None
    if ptype == "binary":
        signals.append(f"`{target}` has two values, so this is a yes/no prediction")
        alternatives.append("Regression on a probability score: rejected because the outcome is a category.")
    elif ptype == "multiclass":
        signals.append(f"`{target}` has {col['unique']} categories to choose between")
        alternatives.append("One binary model per category: rejected; a single multiclass model is simpler and calibrated across classes.")
    else:
        signals.append(f"`{target}` is a continuous number")
        alternatives.append("Classification into ranges: rejected because it throws away precision.")
    if time_col:
        alternatives.append(
            f"Time-series forecasting over `{time_col}`: rejected because the goal is about individual rows, not totals over time."
        )
    signals.append(f"{profile['rows']:,} labelled rows are available")
    conf = 0.92 if score >= 2 else 0.8 if score >= 1 else 0.6
    return Detection(
        problem_type=ptype,
        target=target,
        time_column=time_col,
        entity_column=entity_column(profile),
        positive_class=positive_class(df, target) if ptype == "binary" else None,
        confidence=conf,
        signals=signals,
        alternatives=alternatives,
        candidate_targets=cands[:5],
        leakage_suspects=name_leakage_suspects(profile, target),
    )
