from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from datalyon.agent.events import to_jsonable
from datalyon.agent.llm import make_llm
from datalyon.agent.llm.base import FramingOut
from datalyon.agent.llm.gemini import GeminiLLM
from datalyon.agent.llm.mock import MockLLM
from datalyon.agent.privacy import safe_summary, summary_text
from datalyon.agent.prompts import render
from datalyon.agent.stages.detect import merge
from datalyon.core.config import Settings
from datalyon.ml.detection import detect
from datalyon.ml.profiling import profile_dataframe
from tests.unit.data import churn_df


def test_privacy_masks_personal_data_and_limits_rows() -> None:
    df = churn_df(50).assign(email=[f"user{i}@corp.com" for i in range(50)], note=["call +1 555 123 4567"] * 50)
    prof = profile_dataframe(df)
    summ = safe_summary(df, prof, sample_rows=5)
    assert len(summ["sample_rows"]) == 5
    flat = str(summ)
    assert "@corp.com" not in flat and "555 123" not in flat and "C000" not in flat  # emails, phones, ids masked
    assert any(c.get("masked") for c in summ["columns"])
    assert "rows" in summary_text(summ)


def test_mock_llm_rules() -> None:
    llm = MockLLM()
    assert llm.frame_problem("x", "", {}) is None
    feats = llm.suggest_features("g", "target", ["monthly_fee", "seats", "target"], "")
    assert feats and feats[0]["op"] == "ratio" and feats[0]["a"] == "monthly_fee"
    opts = [{"label": "Yes, use churned", "value": "churned"}, {"label": "Use plan instead", "value": "plan"}]
    assert llm.interpret_reply("q", opts, "yes please", []).choice_index == 0
    assert llm.interpret_reply("q", opts, "use plan instead", []).choice_index == 1
    assert llm.interpret_reply("q", opts, "the nps column", ["nps", "plan"]).column == "nps"
    assert llm.interpret_reply("q", [{"label": "8 weeks", "value": 8}], "12", []).number == 12
    assert llm.interpret_reply("q", opts, "why do you think so?", []).choice_index is None
    expl = llm.explain_model({"algorithm": "LightGBM", "metric": "ROC-AUC", "score": "0.9", "baseline": "0.5", "top_features": ["a", "b"]})
    assert "LightGBM" in expl.why_model and "a and b" in expl.explanation
    assert "drivers" in llm.answer("which features matter?", {"top_features": [["a", 0.5]]})


def test_make_llm_respects_settings() -> None:
    assert make_llm(Settings(llm_provider="gemini", gemini_api_key="")).name == "mock"
    assert make_llm(Settings(llm_provider="mock", gemini_api_key="k")).name == "mock"


class FakeModels:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def generate_content(self, model: str, contents: str, config: Any) -> Any:
        self.prompts.append(contents)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=r)

    def list(self) -> list[Any]:
        return [
            SimpleNamespace(name="models/gemini-9-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-9-flash-image"),
        ]


def _gemini(replies: list[Any]) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(replies)
    return GeminiLLM("key", "gemini-x", client=SimpleNamespace(models=models)), models


def test_gemini_structured_output_is_validated() -> None:
    llm, models = _gemini(['{"problem_type": "binary", "target": "churned", "confidence": 0.9, "signals": ["s"], "title": "Churn risk"}'])
    out = llm.frame_problem("Predict churn", "summary", {"target": None})
    assert isinstance(out, FramingOut) and out.target == "churned" and out.title == "Churn risk"
    assert "<data>Predict churn</data>" in models.prompts[0]


def test_gemini_falls_back_to_rules_on_bad_output_and_errors() -> None:
    llm, _ = _gemini(
        ['{"text": 42, "oops": true}', RuntimeError("500 internal"), RuntimeError("500 internal"), RuntimeError("500 internal")]
    )
    facts = {"reasons": ["r1", "r2"]}
    assert llm.explain_strategy(facts) == "r1 r2"  # invalid JSON -> mock
    assert llm.explain_strategy(facts) == "r1 r2"  # transient errors exhausted -> mock
    assert llm.failures == 2


def test_gemini_switches_model_when_not_found() -> None:
    llm, _ = _gemini([RuntimeError("404 model not found"), '{"text": "ok"}'])
    assert llm.explain_strategy({"reasons": []}) == "ok"
    assert llm.model == "gemini-9-flash"


def test_merge_validates_llm_framing_against_data() -> None:
    df = churn_df(200)
    prof = profile_dataframe(df)
    det = detect("Predict churn", prof, df)
    bogus = FramingOut(problem_type="regression", target="not_a_column", confidence=0.95)
    kept, _, _ = merge(det, bogus, prof, df)
    assert kept.target == "churned"
    wrong_type = FramingOut(
        problem_type="regression",
        target="churned",
        confidence=0.9,
        leakage_suspects=[{"column": "cancel_reason", "reason": "after the fact"}],
    )  # type: ignore[list-item]
    fixed, _, _ = merge(det, wrong_type, prof, df)
    assert fixed.problem_type == "binary"  # the data decides
    assert any(s["column"] == "cancel_reason" for s in fixed.leakage_suspects)


def test_prompts_and_json_helpers() -> None:
    assert "Goal: <data>X</data>" in render("frame_problem.v1", goal="X", summary="", heuristic="{}")
    import numpy as np

    assert to_jsonable({"a": np.float64(1.5), "b": float("nan"), "c": pd.Timestamp("2024-01-02"), "d": np.arange(2)}) == {
        "a": 1.5,
        "b": None,
        "c": "2024-01-02T00:00:00",
        "d": [0, 1],
    }


@pytest.mark.parametrize("text", ["ok", "Looks good"])
def test_yes_detection(text: str) -> None:
    from datalyon.agent.context import StageContext

    assert StageContext._yes(text)
