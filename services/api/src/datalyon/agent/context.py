"""StageContext: what every agent stage uses to talk to the user, persist progress and pause for approvals."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from ..core.config import Settings
from ..db.models import AgentSession
from ..db.repositories.datasets import DatasetRepo
from ..db.repositories.models import ModelRepo
from ..db.session import Database
from ..storage.file_store import LocalFileStore
from .events import EventBus, to_jsonable
from .llm.base import LLM

log = logging.getLogger("datalyon.agent")


class CancelledError(Exception):
    pass


class StopRunError(Exception):
    """An honest stop (e.g. no model beats the baseline): reported to the user, not retryable as-is."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UserFacingError(Exception):
    """A problem the user can fix (bad file, unusable column...). Retryable after a change."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Tool:
    def __init__(self, ctx: StageContext, label: str) -> None:
        self.ctx, self.label, self.id = ctx, label, f"t_{uuid.uuid4().hex[:8]}"
        self.t0 = time.perf_counter()
        self._last = 0.0

    def log(self, line: str, throttle: bool = False) -> None:
        now = time.perf_counter()
        if throttle and now - self._last < 0.25:
            return
        self._last = now
        self.ctx.emit("tool.log", id=self.id, line=str(line)[:300])


class StageContext:
    def __init__(
        self,
        session_id: str,
        goal: str,
        settings: Settings,
        db: Database,
        store: LocalFileStore,
        bus: EventBus,
        llm: LLM,
        state: dict[str, Any],
    ) -> None:
        self.session_id = session_id
        self.goal = goal
        self.settings = settings
        self.db = db
        self.store = store
        self.bus = bus
        self.llm = llm
        self.state = state
        self.mem: dict[str, Any] = {}
        self.cancel_event = threading.Event()
        self.datasets = DatasetRepo(db, store, settings)
        self.models = ModelRepo(db)
        self._reply_event = threading.Event()
        self._reply: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self.n_jobs = max(1, min(4, (__import__("os").cpu_count() or 2) - 1))

    # ---- events ---------------------------------------------------------------------------
    def emit(self, type_: str, **payload: Any) -> None:
        self.bus.emit(self.session_id, type_, **payload)

    def thinking(self, text: str | None) -> None:
        self.emit("thinking", text=text)

    def say(self, text: str, stream: bool = True) -> None:
        mid = f"m_{uuid.uuid4().hex[:8]}"
        self.emit("thinking", text=None)
        if not stream or len(text) < 80:
            self.emit("message", id=mid, text=text)
            return
        self.emit("message", id=mid, text="")
        for i in range(0, len(text), 64):
            self.emit("message.delta", id=mid, delta=text[i : i + 64])
            time.sleep(0.012)

    @contextmanager
    def tool(self, label: str) -> Iterator[Tool]:
        t = Tool(self, label)
        self.emit("tool.start", id=t.id, label=label)
        status = "done"
        try:
            yield t
        except BaseException:
            status = "error"
            raise
        finally:
            self.emit("tool.end", id=t.id, duration_ms=int((time.perf_counter() - t.t0) * 1000), status=status)

    def card(self, kind: str, data: dict[str, Any], id_: str | None = None) -> str:
        cid = id_ or f"{kind}_{uuid.uuid4().hex[:6]}"
        self.emit("card", id=cid, kind=kind, data=to_jsonable(data))
        return cid

    def card_update(self, cid: str, data: dict[str, Any]) -> None:
        self.emit("card.update", id=cid, data=to_jsonable(data))

    def blueprint(self, field: str, value: Any) -> None:
        self.state.setdefault("blueprint", {})[field] = str(value)
        self.emit("blueprint", field=field, value=str(value))

    def decision(self, step: str, decision: str, reason: str, alternatives: str = "") -> None:
        d = {"step": step, "decision": decision, "reason": reason, "alternatives": alternatives}
        self.state.setdefault("decisions", []).append(d)
        self.emit("decision", **d)

    def model_update(self, **fields: Any) -> None:
        mid = self.state.get("model_id")
        if not mid:
            return
        m = self.models.update(mid, **fields)
        self.emit("model", id=m.id, status=m.status, progress=m.progress, stage=m.stage, name=m.name)

    # ---- control --------------------------------------------------------------------------
    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError()

    def checkpoint(self) -> None:
        with self.db.session() as s:
            row = s.get(AgentSession, self.session_id)
            if row is not None:
                row.state = to_jsonable(self.state)

    def set_status(self, status: str, **fields: Any) -> None:
        with self.db.session() as s:
            row = s.get(AgentSession, self.session_id)
            if row is not None:
                row.status = status
                for k, v in fields.items():
                    setattr(row, k, v)
        self.emit("session", status=status)

    # ---- approvals ----------------------------------------------------------------------------
    def ask(
        self,
        kind: str,
        prompt: str,
        options: list[dict[str, Any]] | None = None,
        items: list[dict[str, Any]] | None = None,
        default: Any = None,
    ) -> dict[str, Any]:
        """Pause the run until the user replies. Returns {"value", "text", "label"}."""
        options = options or []
        while True:
            self.check_cancel()
            ask_id = f"a_{uuid.uuid4().hex[:8]}"
            self._pending = {"ask_id": ask_id, "kind": kind, "prompt": prompt, "options": options, "items": items or [], "default": default}
            self.state["pending_ask"] = self._pending
            self.checkpoint()
            self._reply_event.clear()
            self.set_status("waiting")
            self.emit("ask", **self._pending)
            got = self._reply_event.wait(timeout=self.settings.ask_timeout_s)
            self.check_cancel()
            reply: dict[str, Any] = self._reply if (got and self._reply is not None) else {"value": default, "text": None}
            self._reply = None
            self.state.pop("pending_ask", None)
            self.set_status("running")
            value, text = reply.get("value"), reply.get("text")
            if value is not None or not text:
                label = text or self._label_for(options, value)
                self.emit("reply", ask_id=ask_id, text=label, value=value)
                return {"value": value, "text": text, "label": label}
            # free text: map to an option, or answer the question and ask again
            self.emit("reply", ask_id=ask_id, text=text, value=None)
            interp = self.llm.interpret_reply(prompt, options, text, self.mem.get("columns", []))
            if interp.choice_index is not None and 0 <= interp.choice_index < len(options):
                return {"value": options[interp.choice_index]["value"], "text": text, "label": text}
            if interp.column:
                return {"value": interp.column, "text": text, "label": text, "free_column": True}
            if interp.number is not None and any(isinstance(o.get("value"), (int, float)) for o in options):
                return {"value": int(interp.number), "text": text, "label": text}
            if kind == "approval" and items is not None and interp.choice_index is None and not interp.answer and self._yes(text):
                return {"value": default, "text": text, "label": text}
            answer = interp.answer or self.llm.answer(text, self.facts())
            self.say(answer + "\n\nTo continue, please pick one of the options above or type your choice.")

    @staticmethod
    def _yes(text: str) -> bool:
        return text.strip().lower().startswith(("y", "ok", "sure", "go", "approve", "accept", "looks good", "fine"))

    @staticmethod
    def _label_for(options: list[dict[str, Any]], value: Any) -> str:
        for o in options:
            if o.get("value") == value:
                return str(o["label"])
        return str(value) if value is not None else "Continue"

    def deliver_reply(self, ask_id: str, value: Any, text: str | None) -> bool:
        if not self._pending or self._pending["ask_id"] != ask_id:
            return False
        self._reply = {"value": value, "text": text}
        self._reply_event.set()
        return True

    @property
    def pending(self) -> dict[str, Any] | None:
        return self._pending if not self._reply_event.is_set() else None

    def facts(self) -> dict[str, Any]:
        """Compact, numbers-only context for answering questions (no raw rows)."""
        s = self.state
        final = s.get("final") or {}
        return {
            "goal": self.goal,
            "target": (s.get("plan") or {}).get("target"),
            "problem_label": (s.get("plan") or {}).get("problem_label"),
            "metrics": final.get("metrics", []),
            "top_features": final.get("importance", [])[:8],
            "why_model": final.get("why_model"),
            "algorithm": final.get("algorithm"),
            "decisions": s.get("decisions", [])[-12:],
            "critic": [f"{f['severity']}: {f['title']}" for f in (s.get("critic") or {}).get("findings", [])],
        }

    # ---- data -----------------------------------------------------------------------------
    def df(self) -> pd.DataFrame:
        if "df" not in self.mem:
            self.mem["df"] = self.datasets.load_df(self.state["dataset_id"])
            self.mem["columns"] = [str(c) for c in self.mem["df"].columns]
        df: pd.DataFrame = self.mem["df"]
        return df

    def profile(self) -> dict[str, Any]:
        if "profile" not in self.mem:
            ds = self.datasets.get(self.state["dataset_id"])
            self.mem["profile"] = ds.profile or {}
        p: dict[str, Any] = self.mem["profile"]
        return p
