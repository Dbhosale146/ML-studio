import io
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.fixtures.builders import messy_workbook, simple_workbook


def _upload(client: TestClient, path: Path) -> dict[str, Any]:
    r = client.post("/api/v1/datasets", files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
    assert r.status_code == 201, r.text
    return r.json()


def test_single_sheet_upload_is_ready_with_profile(client: TestClient, tmp_path: Path) -> None:
    ds = _upload(client, simple_workbook(tmp_path / "s.xlsx"))
    assert ds["status"] == "ready" and ds["sheet"] == "Sheet1" and ds["rows"] == 40
    prof = client.get(f"/api/v1/datasets/{ds['id']}/profile").json()
    assert prof["rows"] == 40 and {c["name"] for c in prof["columns"]} >= {"age", "churned"}
    prev = client.get(f"/api/v1/datasets/{ds['id']}/preview?limit=5").json()
    assert len(prev["rows"]) == 5
    assert client.get("/api/v1/datasets").json()["total"] == 1
    acts = client.get("/api/v1/activity").json()
    assert any("Uploaded" in a["text"] for a in acts)


def test_multi_sheet_needs_choice_then_profiles(client: TestClient, tmp_path: Path) -> None:
    ds = _upload(client, messy_workbook(tmp_path / "m.xlsx"))
    assert ds["needs_sheet"] and len(ds["sheets"]) == 2
    assert client.get(f"/api/v1/datasets/{ds['id']}/profile").status_code == 409
    chosen = client.post(f"/api/v1/datasets/{ds['id']}/sheet", json={"sheet": "Data"}).json()
    assert chosen["status"] == "ready" and chosen["rows"] == 5
    bad = client.post(f"/api/v1/datasets/{ds['id']}/sheet", json={"sheet": "Nope"})
    assert bad.status_code == 404 and bad.json()["error"]["code"] == "unknown_sheet"


def test_rejects_wrong_content(client: TestClient) -> None:
    r = client.post("/api/v1/datasets", files={"file": ("evil.xlsx", io.BytesIO(b"not really excel"), "application/octet-stream")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "invalid_excel"


def test_delete_dataset(client: TestClient, tmp_path: Path) -> None:
    ds = _upload(client, simple_workbook(tmp_path / "s.xlsx"))
    assert client.delete(f"/api/v1/datasets/{ds['id']}").status_code == 204
    assert client.get(f"/api/v1/datasets/{ds['id']}").status_code == 404
