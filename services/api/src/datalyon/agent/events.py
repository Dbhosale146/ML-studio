"""Typed event emitter: validates each event, persists it (so sessions can be replayed) and keeps an
in-memory tail for live Server-Sent Event streams."""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import func, select

from ..db.models import SessionEvent
from ..db.session import Database
from ..schemas.events import EVENT_MODELS


def to_jsonable(obj: Any) -> Any:
    """numpy / pandas / datetime values -> plain JSON values (NaN and inf become null)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return str(obj)


class EventBus:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._tail: dict[str, list[dict[str, Any]]] = {}

    def _next_seq(self, session_id: str) -> int:
        if session_id not in self._seq:
            with self.db.session() as s:
                cur = s.scalar(select(func.max(SessionEvent.seq)).where(SessionEvent.session_id == session_id))
            self._seq[session_id] = int(cur or 0)
        self._seq[session_id] += 1
        return self._seq[session_id]

    def emit(self, session_id: str, type_: str, **payload: Any) -> dict[str, Any]:
        model = EVENT_MODELS.get(type_)
        if model is None:
            raise ValueError(f"unknown event type {type_}")
        data = to_jsonable(model.model_validate(to_jsonable(payload)).model_dump(mode="json"))
        json.dumps(data)  # fail fast on anything not serialisable
        with self._lock:
            seq = self._next_seq(session_id)
            ev = {"seq": seq, "type": type_, "at": time.time(), "payload": data}
            with self.db.session() as s:
                s.add(SessionEvent(session_id=session_id, seq=seq, type=type_, payload=data))
            tail = self._tail.setdefault(session_id, [])
            tail.append(ev)
            if len(tail) > 5000:
                del tail[:1000]
        return ev

    def after(self, session_id: str, last_seq: int) -> list[dict[str, Any]]:
        """Events with seq > last_seq: from memory when possible, otherwise from the database (replay)."""
        with self._lock:
            tail = self._tail.get(session_id)
            if tail and tail[0]["seq"] <= last_seq + 1:
                return [e for e in tail if e["seq"] > last_seq]
        with self.db.session() as s:
            rows = s.scalars(
                select(SessionEvent).where(SessionEvent.session_id == session_id, SessionEvent.seq > last_seq).order_by(SessionEvent.seq)
            )
            return [{"seq": r.seq, "type": r.type, "at": r.created_at.timestamp(), "payload": r.payload} for r in rows]
