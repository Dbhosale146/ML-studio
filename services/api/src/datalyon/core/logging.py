"""Structured JSON logging with request and session IDs. Never log raw data rows."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        sid = getattr(record, "session_id", None) or session_id_var.get()
        if rid:
            payload["request_id"] = rid
        if sid:
            payload["session_id"] = sid
        for key in ("model_id", "dataset_id", "stage", "duration_ms", "status_code", "path", "method"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers[:] = [handler]
        lg.propagate = False
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our request log middleware
    for noisy in ("httpx", "alembic", "mlflow", "optuna", "shap", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
