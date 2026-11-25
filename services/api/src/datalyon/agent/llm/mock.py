"""MockLLM: deterministic rules and templates. The whole app runs offline with it, and tests use it."""

from __future__ import annotations

import re
from typing import Any

from .base import LLM, ModelExplanation, ReplyInterpretation

MONEY = re.compile(r"amount|fee|price|cost|spend|revenue|total|value|income|salary|balance|sales", re.I)
COUNT = re.compile(
    r"seats|count|qty|quantity|units|sessions|visits|orders|area|sqft|size|rooms|bedrooms|users|days|months|years|tenure|age", re.I
)
YES = re.compile(r"^(y|yes|yeah|yep|ok|okay|sure|go|go ahead|correct|right|sounds good|do it|fine|accept|approve|looks good|lgtm)\b", re.I)
NO = re.compile(r"^(n|no|nope|don't|do not|keep|skip)\b", re.I)
QUESTION = re.compile(r"\?$|^(why|what|how|which|who|when|where|can|could|does|is|are|explain)\b", re.I)


def pretty(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"[_\-.]+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else str(name)


class MockLLM(LLM):
    name = "mock"

    def frame_problem(self, goal: str, summary: str, heuristic: dict[str, Any]) -> None:
        return None  # the heuristic detection is used as-is

    def explain_strategy(self, facts: dict[str, Any]) -> str:
        return " ".join(facts.get("reasons", []))

    def suggest_features(self, goal: str, target: str | None, numeric_columns: list[str], summary: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        money = [c for c in numeric_columns if MONEY.search(c) and c != target]
        counts = [c for c in numeric_columns if COUNT.search(c) and c != target]
        for a in money:
            for b in counts:
                if a != b and len(out) < 3:
                    out.append(
                        {
                            "name": f"{a}_per_{b}",
                            "op": "ratio",
                            "a": a,
                            "b": b,
                            "why": f"{pretty(a)} relative to {pretty(b).lower()} often carries more signal than either alone",
                            "source": "rules",
                        }
                    )
        return out

    def narrate_critic(self, facts: dict[str, Any]) -> str:
        issues = [f for f in facts["findings"] if f["severity"] in ("high", "medium", "low")]
        leader = f"{facts['leader']} ({facts['metric']} {facts['score']})"
        if any(f["check"] == "baseline" and f["severity"] == "high" for f in issues):
            return f"I checked round 1 carefully: even the best model, {leader}, does not clearly beat a naive baseline."
        if not issues:
            return f"Round 1 looks healthy: {leader} leads, with no leakage, overfitting or calibration problems. I'll use round 2 to fine-tune the best candidates."
        if any(f["check"] == "leakage" and f["severity"] == "high" for f in issues):
            return f"{leader} looks too good to be true. I investigated before accepting it and found {len(issues)} thing{'s' if len(issues) > 1 else ''} to fix in round 2."
        return f"Before accepting {leader}, I reviewed the results and found {len(issues)} thing{'s' if len(issues) > 1 else ''} to improve in round 2."

    def explain_model(self, facts: dict[str, Any]) -> ModelExplanation:
        top = [pretty(n).lower() for n in facts.get("top_features", [])[:3]]
        drivers = (", ".join(top[:-1]) + " and " + top[-1]) if len(top) > 1 else (top[0] if top else "")
        what = f"The predictions are driven mostly by {drivers}. " if drivers else ""
        if facts.get("horizon"):
            acc = f"Across the backtest windows its {facts['metric']} was {facts['score']}, against {facts['baseline']} for seasonal naive."
        else:
            acc = f"On rows the model never saw during training, it scored {facts['metric']} {facts['score']} against {facts['baseline']} for a naive baseline."
        why = f"{facts['algorithm']} had the best holdout {facts['metric']} ({facts['score']})"
        if facts.get("runner_up"):
            why += f", ahead of {facts['runner_up']} ({facts['runner_up_score']})"
        why += f". The naive baseline scores {facts['baseline']} on the same rows."
        return ModelExplanation(explanation=what + acc, why_model=why)

    def field_labels(self, columns: list[str], goal: str, target: str | None) -> dict[str, dict[str, str]]:
        return {c: {"label": pretty(c), "help": ""} for c in columns}

    def interpret_reply(self, prompt: str, options: list[dict[str, Any]], text: str, columns: list[str]) -> ReplyInterpretation:
        t = text.strip().lower()
        for n in sorted(columns, key=len, reverse=True):
            if len(n) > 2 and n.lower() in t:
                for i, o in enumerate(options):
                    if str(o.get("value")).lower() == n.lower():
                        return ReplyInterpretation(choice_index=i)
                return ReplyInterpretation(column=n)
        for i, o in enumerate(options):
            lab = re.sub(r"[`*]", "", str(o["label"])).lower()
            if t == lab or (len(t) > 2 and t in lab):
                return ReplyInterpretation(choice_index=i)
        num = re.search(r"\d+(\.\d+)?", t)
        if num and any(isinstance(o.get("value"), (int, float)) for o in options):
            return ReplyInterpretation(number=float(num.group()))
        if QUESTION.search(t):
            return ReplyInterpretation(answer=None)
        if YES.match(t) and options:
            return ReplyInterpretation(choice_index=0)
        if NO.match(t) and len(options) > 1:
            return ReplyInterpretation(choice_index=len(options) - 1)
        return ReplyInterpretation()

    def answer(self, question: str, context: dict[str, Any]) -> str:
        q = question.lower()
        parts: list[str] = []
        if context.get("top_features") and re.search(r"feature|driver|important|matter|why|influen|factor", q):
            parts.append(
                "The strongest drivers are " + ", ".join(f"{pretty(n)} ({v * 100:.0f}%)" for n, v in context["top_features"][:5]) + "."
            )
        if context.get("metrics") and re.search(r"accura|metric|score|good|perform|error|precision|recall", q):
            parts.append("Holdout results: " + ", ".join(f"{m['name']} {m['display']}" for m in context["metrics"][:5]) + ".")
        if context.get("why_model") and re.search(r"why|model|algorithm|chose|choose|pick", q):
            parts.append(str(context["why_model"]))
        if context.get("target") and re.search(r"target|predict|outcome", q):
            parts.append(f"I'm predicting `{context['target']}` as a {context.get('problem_label', 'model')}.")
        if not parts:
            parts.append(
                "I'm running with the offline rule-based assistant (no Gemini key), so I can only answer from the facts of this run: "
                "try asking about the drivers, the metrics, the target or why this model was chosen."
            )
        return " ".join(parts)
