"""Google Gemini provider (google-genai SDK) with structured JSON output validated by Pydantic.

Every method falls back to the MockLLM behaviour when a call fails, times out or returns output that
does not validate, so a flaky API never breaks a training run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..prompts import render
from .base import FeatureIdeas, FieldLabels, FramingOut, ModelExplanation, ReplyInterpretation, TextOut
from .mock import MockLLM

log = logging.getLogger("datalyon.llm")
T = TypeVar("T", bound=BaseModel)

SYSTEM = render("system.v1")


def _transient(msg: str) -> bool:
    m = msg.lower()
    return any(
        t in m for t in ("429", "500", "502", "503", "504", "unavailable", "resource_exhausted", "deadline", "timeout", "temporarily")
    )


def _not_found(msg: str) -> bool:
    m = msg.lower()
    return "404" in m or "not found" in m or "not_found" in m or "is not supported" in m


class GeminiLLM(MockLLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: int = 60, client: Any = None) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._switched = False
        if client is not None:
            self.client = client
        else:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))

    # ---- transport ---------------------------------------------------------------------
    def _switch_model(self) -> bool:
        """The configured model id is not available: pick a current Flash model from the account's list."""
        if self._switched:
            return False
        self._switched = True
        try:
            names = []
            for m in self.client.models.list():
                name = (getattr(m, "name", "") or "").replace("models/", "")
                actions = getattr(m, "supported_actions", None) or []
                if (
                    "gemini" in name
                    and (not actions or "generateContent" in actions)
                    and not any(x in name for x in ("image", "tts", "audio", "embedding", "live"))
                ):
                    names.append(name)
            flash = sorted([n for n in names if "flash" in n and "lite" not in n] or names, reverse=True)
            if flash:
                log.warning("Gemini model %s unavailable, switching to %s", self.model, flash[0])
                self.model = flash[0]
                return True
        except Exception as e:  # pragma: no cover - network dependent
            log.warning("Could not list Gemini models: %s", e.__class__.__name__)
        return False

    def _structured(self, prompt: str, schema: type[T], temperature: float = 0.2) -> T | None:
        from google.genai import types

        for attempt in range(3):
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                )
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                with self._lock:
                    self.calls += 1
                return schema.model_validate_json(resp.text or "")
            except ValidationError as e:
                self.failures += 1
                self.last_error = f"invalid structured output: {e.error_count()} errors"
                log.warning("Gemini returned output that does not match %s", schema.__name__)
                return None
            except Exception as e:
                msg = str(e)
                self.last_error = msg[:300]
                if _not_found(msg) and self._switch_model():
                    continue
                if _transient(msg) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                self.failures += 1
                log.warning("Gemini call failed: %s", e.__class__.__name__)
                return None
        return None

    # ---- tasks (each falls back to the MockLLM rules) ------------------------------------
    def frame_problem(self, goal: str, summary: str, heuristic: dict[str, Any]) -> FramingOut | None:  # type: ignore[override]
        return self._structured(
            render("frame_problem.v1", goal=goal, summary=summary, heuristic=json.dumps(heuristic, default=str)), FramingOut
        )

    def explain_strategy(self, facts: dict[str, Any]) -> str:
        out = self._structured(render("explain_strategy.v1", facts=json.dumps(facts, default=str)), TextOut, 0.3)
        return out.text if out else super().explain_strategy(facts)

    def suggest_features(self, goal: str, target: str | None, numeric_columns: list[str], summary: str) -> list[dict[str, Any]]:
        if len(numeric_columns) < 2:
            return []
        out = self._structured(
            render("suggest_features.v1", goal=goal, target=target or "", columns=", ".join(numeric_columns[:60]), summary=summary),
            FeatureIdeas,
            0.3,
        )
        if out is None:
            return super().suggest_features(goal, target, numeric_columns, summary)
        return [{**f.model_dump(), "source": "llm"} for f in out.features]

    def narrate_critic(self, facts: dict[str, Any]) -> str:
        out = self._structured(render("narrate_critic.v1", facts=json.dumps(facts, default=str)), TextOut, 0.3)
        return out.text if out else super().narrate_critic(facts)

    def explain_model(self, facts: dict[str, Any]) -> ModelExplanation:
        out = self._structured(render("explain_model.v1", facts=json.dumps(facts, default=str)), ModelExplanation, 0.3)
        return out or super().explain_model(facts)

    def field_labels(self, columns: list[str], goal: str, target: str | None) -> dict[str, dict[str, str]]:
        base = super().field_labels(columns, goal, target)
        out = self._structured(render("field_labels.v1", goal=goal, target=target or "", columns=", ".join(columns[:80])), FieldLabels)
        if out:
            for c, v in out.fields.items():
                if c in base:
                    base[c] = {"label": v.label[:40], "help": v.help[:140]}
        return base

    def interpret_reply(self, prompt: str, options: list[dict[str, Any]], text: str, columns: list[str]) -> ReplyInterpretation:
        quick = super().interpret_reply(prompt, options, text, columns)
        if quick.choice_index is not None or quick.column:
            return quick
        out = self._structured(
            render(
                "interpret_reply.v1",
                prompt=prompt,
                options=json.dumps([{"index": i, "label": o["label"]} for i, o in enumerate(options)]),
                text=text,
                columns=", ".join(columns[:80]),
            ),
            ReplyInterpretation,
        )
        return out or quick

    def answer(self, question: str, context: dict[str, Any]) -> str:
        out = self._structured(render("answer.v1", question=question, context=json.dumps(context, default=str)[:12000]), TextOut, 0.3)
        return out.text if out else super().answer(question, context)
