"""LLM interface and structured output schemas. The LLM reasons and writes; it never computes a metric."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LeakSuspect(BaseModel):
    column: str
    reason: str


class FramingOut(BaseModel):
    problem_type: Literal["binary", "multiclass", "regression", "forecasting"] | None = None
    target: str | None = None
    time_column: str | None = None
    series_column: str | None = None
    positive_class: str | None = None
    horizon: int | None = None
    confidence: float = Field(0.5, ge=0, le=1)
    signals: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    title: str | None = None
    leakage_suspects: list[LeakSuspect] = Field(default_factory=list)
    clarifying_question: str | None = None


class TextOut(BaseModel):
    text: str


class FeatureIdea(BaseModel):
    name: str
    op: Literal["ratio", "diff", "sum", "product", "log1p"]
    a: str
    b: str | None = None
    why: str = ""


class FeatureIdeas(BaseModel):
    features: list[FeatureIdea] = Field(default_factory=list)


class ModelExplanation(BaseModel):
    explanation: str
    why_model: str


class FieldLabel(BaseModel):
    label: str
    help: str = ""


class FieldLabels(BaseModel):
    fields: dict[str, FieldLabel] = Field(default_factory=dict)


class ReplyInterpretation(BaseModel):
    choice_index: int | None = None
    column: str | None = None
    number: float | None = None
    answer: str | None = None  # the user asked something instead of choosing: answer it, then ask again


class LLM:
    """Interface implemented by MockLLM (rules + templates) and GeminiLLM."""

    name: str = "base"
    model: str | None = None

    def frame_problem(self, goal: str, summary: str, heuristic: dict[str, Any]) -> FramingOut | None:
        raise NotImplementedError

    def explain_strategy(self, facts: dict[str, Any]) -> str:
        raise NotImplementedError

    def suggest_features(self, goal: str, target: str | None, numeric_columns: list[str], summary: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def narrate_critic(self, facts: dict[str, Any]) -> str:
        raise NotImplementedError

    def explain_model(self, facts: dict[str, Any]) -> ModelExplanation:
        raise NotImplementedError

    def field_labels(self, columns: list[str], goal: str, target: str | None) -> dict[str, dict[str, str]]:
        raise NotImplementedError

    def interpret_reply(self, prompt: str, options: list[dict[str, Any]], text: str, columns: list[str]) -> ReplyInterpretation:
        raise NotImplementedError

    def answer(self, question: str, context: dict[str, Any]) -> str:
        raise NotImplementedError
