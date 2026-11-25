"""Application settings. Everything tunable comes from environment variables (documented in .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_ROOT = Path(__file__).resolve().parents[3]  # services/api
REPO_ROOT = API_ROOT.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(REPO_ROOT / ".env", API_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Datalyon AI Data Scientist"
    version: str = "0.1.0"
    display_name: str = "there"
    log_level: str = "INFO"

    llm_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    llm_sample_rows: int = Field(5, ge=0, le=5)
    llm_timeout_s: int = 60

    storage_dir: Path = REPO_ROOT / "storage"
    samples_dir: Path = REPO_ROOT / "data" / "samples"
    database_url: str | None = None

    max_upload_mb: int = 50
    max_rows: int = 500_000
    max_columns: int = 300
    train_time_budget_s: int = 300
    random_state: int = 42
    mlflow_enabled: bool = True
    ask_timeout_s: int = 6 * 3600

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"]

    @property
    def storage(self) -> Path:
        p = self.storage_dir
        return p if p.is_absolute() else (API_ROOT / p).resolve()

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.storage / 'datalyon.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        return self.storage / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.storage / "outputs"

    @property
    def mlruns_uri(self) -> str:
        return f"sqlite:///{(self.storage / 'mlruns' / 'mlflow.db').as_posix()}"

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider == "gemini" and bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
