"""End-to-end agent runs through the HTTP API with the offline MockLLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLES
from tests.fixtures.builders import simple_workbook
from tests.integration.agent_driver import cards, drive, of

pytestmark = pytest.mark.slow


def _sample(client: TestClient, name: str) -> dict[str, Any]:
    if not (SAMPLES / name).exists():
        pytest.skip("run `make samples` first")
    r = client.post(f"/api/v1/datasets/samples/{name}")
    assert r.status_code == 201, r.text
    return r.json()


def test_churn_run_catches_leakage_and_publishes(client: TestClient) -> None:
    ds = _sample(client, "customer_churn.xlsx")
    assert ds["needs_sheet"]
    s = client.post("/api/v1/sessions", json={"goal": "Predict which customers will churn", "dataset_id": ds["id"]}).json()
    events = drive(client, s["id"])
    assert events[-1]["type"] == "done", events[-1]
    asks = [a["kind"] for a in of(events, "ask")]
    assert asks == ["sheet", "options", "approval", "options"]  # sheet, confirm target, cleaning plan, accept final
    stages = [(p["key"], p["status"]) for p in of(events, "stage") if p["status"] == "done"]
    assert [k for k, _ in stages] == ["data", "detect", "strategy", "prep", "train", "critic", "final", "publish"]
    critic = cards(events, "critic")[0]
    assert any(f["check"] == "leakage" and f["evidence"].get("column") == "cancellation_reason" for f in critic["findings"])
    lb1, lb2 = cards(events, "leaderboard")
    assert lb1["done"] and lb2["done"] and any(r.get("prev_display") for r in lb2["rows"])
    final = cards(events, "final")[0]
    assert final["primary"]["key"] == "pr_auc" and final["charts"]["roc"] and final["importance"]
    assert {"blueprint", "decision", "experiment", "model", "tool.log", "message.delta"} <= {e["type"] for e in events}

    model = client.get(f"/api/v1/models/{final and of(events, 'done')[0]['model_id']}").json()
    assert model["status"] == "ready" and model["latest_version"] == 1
    assert model["card"]["data_hash"] and model["card"]["library_versions"]["scikit-learn"]
    assert "cancellation_reason" not in model["card"]["columns"]
    schema = client.get(f"/api/v1/models/{model['id']}/input-schema").json()
    assert schema["fields"] and schema["examples"]
    sess = client.get(f"/api/v1/sessions/{s['id']}").json()
    assert sess["status"] == "done" and sess["model_id"] == model["id"]

    # replay over SSE after the run finished
    with client.stream("GET", f"/api/v1/sessions/{s['id']}/events", headers={"Last-Event-ID": str(events[-5]["seq"])}) as r:
        body = "".join(chunk for chunk in r.iter_text())
    seqs = [json.loads(line[6:])["seq"] for line in body.splitlines() if line.startswith("data: {") and '"seq"' in line]
    assert seqs == [e["seq"] for e in events[-4:]] and "event: end" in body

    # free-text question after the run
    r = client.post(f"/api/v1/sessions/{s['id']}/message", json={"text": "Which features matter most?"})
    assert r.json()["message"] == "answered"
    time.sleep(0.2)
    tail = client.app.state.agent.bus.after(s["id"], events[-1]["seq"])  # type: ignore[attr-defined]
    assert any(e["type"] == "message" and "drivers" in e["payload"]["text"] for e in tail)


def test_forecast_run_asks_horizon(client: TestClient) -> None:
    ds = _sample(client, "store_weekly_demand.xlsx")

    def answer(ask: dict[str, Any]) -> dict[str, Any] | None:
        if "How far ahead" in ask["prompt"]:
            return {"value": None, "text": "4 weeks please"}
        return None

    s = client.post("/api/v1/sessions", json={"goal": "Forecast weekly units sold per store", "dataset_id": ds["id"]}).json()
    events = drive(client, s["id"], answer)
    assert events[-1]["type"] == "done", events[-1]
    final = cards(events, "final")[0]
    assert final["horizon"] == 4 and final["primary"]["key"] in ("mape", "wape")
    model_id = of(events, "done")[0]["model_id"]
    schema = client.get(f"/api/v1/models/{model_id}/input-schema").json()
    assert schema["forecast"]["horizon"] == 4 and "Total" in schema["forecast"]["series"]


def test_upload_ask_free_text_cancel_and_retry(client: TestClient, tmp_path: Path) -> None:
    s = client.post("/api/v1/sessions", json={"goal": "Predict whether a customer churned"}).json()
    bus = client.app.state.agent.bus  # type: ignore[attr-defined]
    deadline = time.time() + 30
    ask = None
    while time.time() < deadline and ask is None:
        ask = next((e["payload"] for e in bus.after(s["id"], 0) if e["type"] == "ask"), None)
        time.sleep(0.1)
    assert ask and ask["kind"] == "upload"
    ds = client.post("/api/v1/datasets", files={"file": ("crm.xlsx", simple_workbook(tmp_path / "crm.xlsx", n=120).read_bytes())}).json()
    assert client.post(f"/api/v1/sessions/{s['id']}/reply", json={"ask_id": ask["ask_id"], "value": ds["id"]}).status_code == 200
    # wait for the target confirmation, answer with free text that names the column
    target_ask = None
    while time.time() < deadline + 30 and target_ask is None:
        target_ask = next((e["payload"] for e in bus.after(s["id"], 0) if e["type"] == "ask" and e["payload"]["kind"] == "options"), None)
        time.sleep(0.1)
    assert target_ask and "churned" in target_ask["prompt"]
    cancelled = client.post(f"/api/v1/sessions/{s['id']}/cancel").json()
    for _ in range(50):
        if client.get(f"/api/v1/sessions/{s['id']}").json()["status"] == "cancelled":
            break
        time.sleep(0.1)
    assert client.get(f"/api/v1/sessions/{s['id']}").json()["status"] == "cancelled", cancelled
    assert client.post(f"/api/v1/sessions/{s['id']}/reply", json={"ask_id": target_ask["ask_id"], "value": "x"}).status_code == 409
    before_retry = bus.after(s["id"], 0)[-1]["seq"]
    assert client.post(f"/api/v1/sessions/{s['id']}/retry").status_code == 200

    def answer(a: dict[str, Any]) -> dict[str, Any] | None:
        if a["kind"] == "options" and "target" in a["prompt"]:
            return {"value": None, "text": "yes, churned is right"}
        return None

    events = drive(client, s["id"], answer, after=before_retry)
    assert events[-1]["type"] in ("done", "error")
    replies = [p["text"] for p in of(events, "reply")]
    assert "yes, churned is right" in replies


def test_no_signal_stops_honestly(client: TestClient, tmp_path: Path) -> None:
    import numpy as np
    from openpyxl import Workbook

    rng = np.random.default_rng(7)
    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b", "c", "outcome"])
    for _ in range(400):
        ws.append([float(rng.normal()), float(rng.normal()), str(rng.choice(["x", "y"])), str(rng.choice(["Yes", "No"]))])
    p = tmp_path / "noise.xlsx"
    wb.save(p)
    ds = client.post("/api/v1/datasets", files={"file": ("noise.xlsx", p.read_bytes())}).json()
    s = client.post("/api/v1/sessions", json={"goal": "Predict the outcome", "dataset_id": ds["id"]}).json()
    events = drive(client, s["id"])
    err = of(events, "error")
    assert err and err[0]["code"] == "no_signal" and err[0]["retryable"] is False
    assert client.post(f"/api/v1/sessions/{s['id']}/retry").status_code == 409
    model_id = client.get(f"/api/v1/sessions/{s['id']}").json()["model_id"]
    assert client.get(f"/api/v1/models/{model_id}").json()["status"] == "failed"
