"""Test helper: drive an agent session through the API, answering each question with a chosen or default reply."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

Answer = Callable[[dict[str, Any]], dict[str, Any] | None]


def drive(client: TestClient, session_id: str, answer: Answer | None = None, timeout: float = 600, after: int = 0) -> list[dict[str, Any]]:
    """Consume events until done/error; reply to asks with answer(ask) or the ask's default."""
    bus = client.app.state.agent.bus  # type: ignore[attr-defined]
    events: list[dict[str, Any]] = []
    last = after
    answered: set[str] = set()
    t0 = time.time()
    while time.time() - t0 < timeout:
        new = bus.after(session_id, last)
        for ev in new:
            last = ev["seq"]
            events.append(ev)
            if ev["type"] in ("done", "error"):
                return events
        pending = [e for e in events if e["type"] == "ask" and e["payload"]["ask_id"] not in answered]
        for ev in pending:
            ask = ev["payload"]
            answered.add(ask["ask_id"])
            body = (answer(ask) if answer else None) or {"value": ask.get("default")}
            r = client.post(f"/api/v1/sessions/{session_id}/reply", json={"ask_id": ask["ask_id"], **body})
            assert r.status_code == 200, r.text
        if not new:
            time.sleep(0.1)
    raise AssertionError(f"session {session_id} did not finish in {timeout}s; last events: {[e['type'] for e in events[-10:]]}")


def of(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [e["payload"] for e in events if e["type"] == type_]


def cards(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Final data of each card of a kind (card + its updates applied)."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for e in events:
        if e["type"] == "card" and e["payload"]["kind"] == kind:
            by_id[e["payload"]["id"]] = e["payload"]["data"]
            order.append(e["payload"]["id"])
        elif e["type"] == "card.update" and e["payload"]["id"] in by_id:
            by_id[e["payload"]["id"]] = e["payload"]["data"]
    return [by_id[i] for i in order]
