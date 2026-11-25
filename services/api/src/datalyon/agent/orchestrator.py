"""Agent orchestrator: runs the stage state machine for each session in the background job runner,
pauses on approvals, supports cancel and retry-from-failed-stage, and recovers after a restart."""

from __future__ import annotations

import contextlib
import logging
import threading
from types import ModuleType
from typing import Any

from fastapi import FastAPI
from sqlalchemy import select

from ..core.config import Settings
from ..core.errors import AppError, NotFoundError
from ..core.ids import new_id
from ..core.logging import session_id_var
from ..db.models import AgentSession, Model
from ..db.session import Database
from ..jobs.runner import ThreadJobRunner
from ..ml.forecasting import ForecastError
from ..ml.ingest import IngestError
from ..ml.training import TrainingError
from ..schemas.events import STAGE_KEYS, STAGES
from ..storage.file_store import LocalFileStore
from .context import CancelledError, StageContext, StopRunError, UserFacingError
from .events import EventBus
from .llm import LLM, make_llm
from .stages import critic, data, detect, final, prep, publish, strategy, train
from .stages.common import ResumeFromError

log = logging.getLogger("datalyon.agent")
STAGE_MODULES: list[ModuleType] = [data, detect, strategy, prep, train, critic, final, publish]
TERMINAL = ("done", "failed", "cancelled")


class AgentRun:
    def __init__(self, service: AgentService, row: AgentSession) -> None:
        self.service = service
        self.session_id = row.id
        self.ctx = StageContext(
            row.id, row.goal, service.settings, service.db, service.store, service.bus, service.llm, dict(row.state or {})
        )
        self.thread_done = threading.Event()

    def start(self, index: int) -> None:
        self.service.runner.submit(f"session-{self.session_id}", lambda: self._execute(index))

    def _fail(self, index: int, code: str, message: str, retryable: bool) -> None:
        ctx = self.ctx
        key, label = STAGES[index]
        ctx.emit("stage", index=index, key=key, label=label, status="failed")
        ctx.emit("thinking", text=None)
        ctx.emit("error", code=code, message=message, retryable=retryable)
        ctx.state["stage_index"] = index
        ctx.checkpoint()
        status = "cancelled" if code == "cancelled" else "failed"
        ctx.set_status(status, error={"code": code, "message": message, "retryable": retryable, "stage": key})
        if ctx.state.get("model_id"):
            with contextlib.suppress(AppError):
                ctx.model_update(status="failed", stage=f"Stopped at {label.lower()}", error=message[:500])

    def _execute(self, index: int) -> None:
        token = session_id_var.set(self.session_id)
        ctx = self.ctx
        i = index
        try:
            while i < len(STAGES):
                key, label = STAGES[i]
                try:
                    ctx.check_cancel()
                    ctx.state["stage_index"] = i
                    ctx.set_status("running", stage_key=key, stage_index=i, error=None)
                    ctx.emit("stage", index=i, key=key, label=label, status="active")
                    log.info("stage start", extra={"stage": key})
                    STAGE_MODULES[i].run(ctx)
                    ctx.emit("stage", index=i, key=key, label=label, status="done")
                    done = ctx.state.setdefault("done_stages", [])
                    if key not in done:
                        done.append(key)
                    ctx.checkpoint()
                    i += 1
                except ResumeFromError as r:
                    j = STAGE_KEYS.index(r.stage)
                    ctx.emit("stage", index=i, key=key, label=label, status="skipped")
                    ctx.say(f"I need to redo the {STAGES[j][1].lower()} step first (its results were not kept), then I'll continue.")
                    i = j
                except CancelledError:
                    self._fail(i, "cancelled", "You cancelled this run.", True)
                    return
                except StopRunError as e:
                    ctx.say(e.message)
                    self._fail(i, e.code, e.message, False)
                    return
                except (UserFacingError, IngestError, TrainingError, ForecastError) as e:
                    msg = getattr(e, "message", None) or str(e)
                    self._fail(i, getattr(e, "code", "invalid_data"), msg, True)
                    return
                except AppError as e:
                    self._fail(i, e.code, e.message, True)
                    return
                except Exception:
                    log.exception("stage crashed", extra={"stage": key})
                    self._fail(
                        i,
                        "internal_error",
                        f"Something went wrong during {label.lower()}. The details are in the server log; you can retry from this step.",
                        True,
                    )
                    return
            ctx.state.pop("pending_ask", None)
            ctx.checkpoint()
            ctx.set_status("done", error=None)
            ctx.emit("done", model_id=ctx.state.get("model_id"))
        finally:
            session_id_var.reset(token)
            self.thread_done.set()
            self.service.finished(self.session_id)


class AgentService:
    def __init__(self, settings: Settings, db: Database, store: LocalFileStore, bus: EventBus, llm: LLM, runner: ThreadJobRunner) -> None:
        self.settings, self.db, self.store, self.bus, self.llm, self.runner = settings, db, store, bus, llm, runner
        self.runs: dict[str, AgentRun] = {}
        self._lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------------------------------
    def recover(self) -> None:
        """Runs that were in progress when the server stopped become failed + retryable."""
        with self.db.session() as s:
            for row in s.scalars(select(AgentSession).where(AgentSession.status.in_(["running", "waiting"]))):
                row.status = "failed"
                row.error = {
                    "code": "interrupted",
                    "message": "The server restarted while this run was in progress. Retry to continue from where it stopped.",
                    "retryable": True,
                    "stage": row.stage_key,
                }
                if row.model_id:
                    m = s.get(Model, row.model_id)
                    if m is not None and m.status in ("training", "draft"):
                        m.status, m.error = "failed", "Interrupted by a server restart"

    def finished(self, session_id: str) -> None:
        with self._lock:
            run = self.runs.get(session_id)
            if run is not None and run.thread_done.is_set():
                self.runs.pop(session_id, None)

    # ---- commands --------------------------------------------------------------------------
    def create(self, goal: str, dataset_id: str | None) -> AgentSession:
        state: dict[str, Any] = {}
        if dataset_id:
            from ..db.repositories.datasets import DatasetRepo

            DatasetRepo(self.db, self.store, self.settings).get(dataset_id)
            state["dataset_id"] = dataset_id
        row = AgentSession(id=new_id("ses"), goal=goal.strip(), dataset_id=dataset_id, status="running", state=state)
        with self.db.session() as s:
            s.add(row)
        run = AgentRun(self, row)
        with self._lock:
            self.runs[row.id] = run
        run.start(0)
        return row

    def get(self, session_id: str) -> AgentSession:
        with self.db.session() as s:
            row = s.get(AgentSession, session_id)
            if row is None:
                raise NotFoundError("Session", session_id)
            return row

    def list(self, limit: int = 20) -> list[AgentSession]:
        with self.db.session() as s:
            return list(s.scalars(select(AgentSession).order_by(AgentSession.created_at.desc()).limit(limit)))

    def reply(self, session_id: str, ask_id: str, value: Any, text: str | None) -> bool:
        self.get(session_id)
        run = self.runs.get(session_id)
        if run is None:
            raise AppError("not_waiting", "This run is not waiting for an answer.", 409)
        if not run.ctx.deliver_reply(ask_id, value, text):
            raise AppError("stale_question", "That question has already been answered or replaced.", 409)
        return True

    def message(self, session_id: str, text: str) -> str:
        row = self.get(session_id)
        run = self.runs.get(session_id)
        if run is not None and run.ctx.pending:
            run.ctx.deliver_reply(run.ctx.pending["ask_id"], None, text)
            return "answered_pending_question"
        self.bus.emit(session_id, "reply", ask_id=None, text=text, value=None)
        ctx = (
            run.ctx
            if run
            else StageContext(row.id, row.goal, self.settings, self.db, self.store, self.bus, self.llm, dict(row.state or {}))
        )
        answer = self.llm.answer(text, ctx.facts())
        self.bus.emit(session_id, "message", id=new_id("m"), text=answer)
        return "answered"

    def cancel(self, session_id: str) -> AgentSession:
        row = self.get(session_id)
        run = self.runs.get(session_id)
        if run is not None:
            run.ctx.cancel_event.set()
            run.ctx._reply_event.set()  # wake a run that is waiting for an answer
        elif row.status not in TERMINAL:
            with self.db.session() as s:
                r = s.get(AgentSession, session_id)
                if r is not None:
                    r.status = "cancelled"
        return self.get(session_id)

    def retry(self, session_id: str) -> AgentSession:
        row = self.get(session_id)
        if session_id in self.runs:
            raise AppError("already_running", "This run is still in progress.", 409)
        if row.status not in ("failed", "cancelled"):
            raise AppError("not_retryable", "Only failed or cancelled runs can be retried.", 409)
        if row.error and row.error.get("retryable") is False:
            raise AppError("not_retryable", "This run stopped for a reason a retry cannot fix. Start a new run instead.", 409)
        index = int((row.state or {}).get("stage_index", row.stage_index or 0))
        run = AgentRun(self, row)
        with self._lock:
            self.runs[session_id] = run
        if run.ctx.state.get("model_id"):
            run.ctx.model_update(status="training" if index >= STAGE_KEYS.index("train") else "draft", error=None, stage="Retrying")
        self.bus.emit(session_id, "message", id=new_id("m"), text=f"Retrying from the {STAGES[index][1].lower()} step.")
        run.start(index)
        return self.get(session_id)

    def shutdown(self) -> None:
        """Stop every active run so worker threads (possibly waiting for an answer) let the process exit."""
        for run in list(self.runs.values()):
            run.ctx.cancel_event.set()
            run.ctx._reply_event.set()
        self.runner.shutdown()

    def status(self, session_id: str) -> str:
        return self.get(session_id).status


def attach_agent(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    bus = EventBus(app.state.db)
    runner = ThreadJobRunner(max_workers=3)
    service = AgentService(settings, app.state.db, app.state.store, bus, make_llm(settings), runner)
    app.state.agent = service
    app.state.bus = bus
    app.state.runner = runner
    app.state.on_startup = service.recover
    app.state.on_shutdown = service.shutdown
