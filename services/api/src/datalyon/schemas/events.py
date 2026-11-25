"""Agent event protocol (see docs/agent-events.md). Every event is validated against these models before it is
persisted and streamed; the frontend mirrors them as Zod schemas in apps/web/src/features/agent/events.ts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

STAGES: list[tuple[str, str]] = [
    ("data", "Data & profile"),
    ("detect", "Problem detection"),
    ("strategy", "ML strategy"),
    ("prep", "Cleaning & features"),
    ("train", "Model search + HPO"),
    ("critic", "Critic review + round 2"),
    ("final", "Evaluation & explain"),
    ("publish", "Publish for testing"),
]
STAGE_KEYS = [k for k, _ in STAGES]


class StageEvent(BaseModel):
    index: int
    key: str
    label: str
    status: Literal["active", "done", "failed", "skipped"]


class ThinkingEvent(BaseModel):
    text: str | None = None  # None clears the indicator


class MessageEvent(BaseModel):
    id: str
    text: str = ""


class MessageDeltaEvent(BaseModel):
    id: str
    delta: str


class ToolStartEvent(BaseModel):
    id: str
    label: str


class ToolLogEvent(BaseModel):
    id: str
    line: str


class ToolEndEvent(BaseModel):
    id: str
    duration_ms: int
    status: Literal["done", "error"] = "done"


CardKind = Literal["detection", "profile", "strategy", "cleaning", "features", "leaderboard", "critic", "final", "published"]


class CardEvent(BaseModel):
    id: str
    kind: CardKind
    data: dict[str, Any]


class CardUpdateEvent(BaseModel):
    id: str
    data: dict[str, Any]


class AskOption(BaseModel):
    label: str
    value: Any = None
    hint: str | None = None


class AskEvent(BaseModel):
    ask_id: str
    kind: Literal["options", "free_text", "approval", "upload", "sheet"]
    prompt: str
    options: list[AskOption] = Field(default_factory=list)
    items: list[dict[str, Any]] = Field(default_factory=list)  # approval checklists (cleaning plan)
    default: Any = None


class ReplyEvent(BaseModel):
    """The user's answer, echoed so a replayed conversation shows both sides."""

    ask_id: str | None = None
    text: str
    value: Any = None


class BlueprintEvent(BaseModel):
    field: str
    value: str


class DecisionEvent(BaseModel):
    step: str
    decision: str
    reason: str
    alternatives: str = ""


class ExperimentEvent(BaseModel):
    run_id: str
    model: str
    round: int
    score: float | None
    display: str
    best: bool = False
    tuned: bool = False


class ModelEvent(BaseModel):
    id: str
    status: Literal["draft", "training", "ready", "failed"]
    progress: int
    stage: str | None = None
    name: str | None = None


class ErrorEvent(BaseModel):
    code: str
    message: str
    retryable: bool = True


class DoneEvent(BaseModel):
    model_id: str | None


class SessionStatusEvent(BaseModel):
    status: Literal["running", "waiting", "done", "failed", "cancelled"]


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "stage": StageEvent,
    "thinking": ThinkingEvent,
    "message": MessageEvent,
    "message.delta": MessageDeltaEvent,
    "tool.start": ToolStartEvent,
    "tool.log": ToolLogEvent,
    "tool.end": ToolEndEvent,
    "card": CardEvent,
    "card.update": CardUpdateEvent,
    "ask": AskEvent,
    "reply": ReplyEvent,
    "blueprint": BlueprintEvent,
    "decision": DecisionEvent,
    "experiment": ExperimentEvent,
    "model": ModelEvent,
    "error": ErrorEvent,
    "done": DoneEvent,
    "session": SessionStatusEvent,
}


class AgentEventEnvelope(BaseModel):
    """What the SSE stream sends in each `data:` line."""

    seq: int
    type: str
    at: float
    payload: dict[str, Any]
