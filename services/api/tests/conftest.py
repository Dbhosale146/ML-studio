"""Test configuration: isolated storage, offline MockLLM, no MLflow, small training budgets."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="datalyon-tests-"))
os.environ.update(
    {
        "STORAGE_DIR": str(_TMP / "storage"),
        "LLM_PROVIDER": "mock",
        "GEMINI_API_KEY": "",
        "MLFLOW_ENABLED": "false",
        "TRAIN_TIME_BUDGET_S": "40",
        "LOG_LEVEL": "WARNING",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from datalyon.core.config import Settings, get_settings  # noqa: E402
from datalyon.main import create_app  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES = REPO_ROOT / "data" / "samples"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    get_settings.cache_clear()
    return Settings(storage_dir=tmp_path / "storage", llm_provider="mock", mlflow_enabled=False, train_time_budget_s=40)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
