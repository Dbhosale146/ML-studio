from fastapi.testclient import TestClient


def test_health_reports_mock_llm_and_limits(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"]["provider"] == "mock"
    assert body["limits"]["max_upload_mb"] == 50
    assert r.headers["X-Request-ID"]


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_openapi_is_served(client: TestClient) -> None:
    r = client.get("/api/v1/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/health" in r.json()["paths"]
