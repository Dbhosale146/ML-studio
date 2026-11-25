"""Playground routes: single prediction with drivers, batch scoring + download, forecast view + download."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from tests.conftest import SAMPLES
from tests.integration.agent_driver import drive, of

pytestmark = pytest.mark.slow


def _regression_book(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(3)
    wb = Workbook()
    ws = wb.active
    ws.append(["listing", "rooms", "area_sqm", "district", "price"])
    for i in range(n):
        rooms = int(rng.integers(1, 6))
        area = float(rng.normal(40 + 18 * rooms, 12))
        district = str(rng.choice(["Centre", "North", "Harbour"]))
        price = 1500 * area + {"Centre": 40000, "North": 0, "Harbour": 20000}[district] + float(rng.normal(0, 15000))
        ws.append([f"L{i:04d}", rooms, round(area, 1), district, round(price)])
    wb.save(path)
    return path


def _train(client: TestClient, path: Path, goal: str) -> str:
    ds = client.post("/api/v1/datasets", files={"file": (path.name, path.read_bytes())}).json()
    s = client.post("/api/v1/sessions", json={"goal": goal, "dataset_id": ds["id"]}).json()
    events = drive(client, s["id"])
    assert events[-1]["type"] == "done", events[-1]
    return str(of(events, "done")[0]["model_id"])


def test_single_and_batch_predictions(client: TestClient, tmp_path: Path) -> None:
    book = _regression_book(tmp_path / "flats.xlsx")
    model_id = _train(client, book, "Predict the sale price of a flat")
    schema = client.get(f"/api/v1/models/{model_id}/input-schema").json()
    names = {f["name"] for f in schema["fields"]}
    assert {"rooms", "area_sqm", "district"} <= names and "price" not in names

    # single row: typed fields, drivers, out-of-range warning
    r = client.post(
        f"/api/v1/models/{model_id}/predict", json={"rows": [{"rooms": 3, "area_sqm": 900, "district": "Centre"}], "explain": True}
    )
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert isinstance(row["prediction"], float) and row["drivers"]
    assert any("outside the training range" in w for w in row["warnings"]), row["warnings"]
    lots = client.post(f"/api/v1/models/{model_id}/predict", json={"rows": [{"rooms": "lots", "colour": "red"}]}).json()["rows"][0]
    assert any("not a number" in w for w in lots["warnings"]) and any("unknown field" in w for w in lots["warnings"])
    assert client.post(f"/api/v1/models/{model_id}/predict", json={"rows": []}).status_code == 422

    # batch: upload rows without the target, poll, download the xlsx
    df = pd.read_excel(book).drop(columns=["price"]).head(25)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    job = client.post(f"/api/v1/models/{model_id}/predict-batch", files={"file": ("new_flats.xlsx", buf.getvalue())})
    assert job.status_code == 202, job.text
    jid = job.json()["id"]
    out: dict[str, Any] = {}
    for _ in range(100):
        out = client.get(f"/api/v1/predictions/{jid}").json()
        if out["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert out["status"] == "done", out
    assert out["rows"] == 25 and out["download_url"]
    dl = client.get(out["download_url"])
    assert dl.status_code == 200 and "new_flats_predictions.xlsx" in dl.headers["content-disposition"]
    scored = pd.read_excel(io.BytesIO(dl.content))
    assert len(scored) == 25 and any("predict" in c.lower() for c in scored.columns)

    assert client.get(f"/api/v1/models/{model_id}").json()["prediction_count"] >= 26
    assert client.get(f"/api/v1/models/{model_id}/forecast/download").status_code == 409
    assert client.get("/api/v1/predictions/pj_missing").status_code == 404
    bad_file = client.post(f"/api/v1/models/{model_id}/predict-batch", files={"file": ("x.pdf", b"%PDF-1.4 nope")})
    assert bad_file.status_code == 415


def test_forecast_view_and_download(client: TestClient) -> None:
    if not (SAMPLES / "store_weekly_demand.xlsx").exists():
        pytest.skip("run `make samples` first")
    ds = client.post("/api/v1/datasets/samples/store_weekly_demand.xlsx").json()
    s = client.post("/api/v1/sessions", json={"goal": "Forecast weekly units sold per store", "dataset_id": ds["id"]}).json()
    events = drive(client, s["id"])
    assert events[-1]["type"] == "done", events[-1]
    model_id = of(events, "done")[0]["model_id"]
    schema = client.get(f"/api/v1/models/{model_id}/input-schema").json()
    series = next(x for x in schema["forecast"]["series"] if x != "Total")
    r = client.post(f"/api/v1/models/{model_id}/predict", json={"series": series, "horizon": 3})
    assert r.status_code == 200, r.text
    view = r.json()["forecast"]
    assert view["series"] == series and len(view["table"]) == 3
    dl = client.get(f"/api/v1/models/{model_id}/forecast/download", params={"series": series, "horizon": 3})
    assert dl.status_code == 200
    assert len(pd.read_excel(io.BytesIO(dl.content))) == 3
    assert client.post(f"/api/v1/models/{model_id}/predict-batch", files={"file": ("a.csv", b"a,b\n1,2\n")}).status_code == 409
