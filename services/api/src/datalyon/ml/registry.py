"""Versioned model artifacts (joblib pipeline + model card + input schema) and MLflow run logging."""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

log = logging.getLogger("datalyon.ml.registry")

LIBS = ("scikit-learn", "lightgbm", "xgboost", "pandas", "numpy", "optuna", "shap", "joblib")


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:  # pragma: no cover
            out[lib] = "not installed"
    return out


def version_dir(root: Path, model_id: str, v: int) -> Path:
    return root / "models" / model_id / f"v{v}"


def save_version(root: Path, model_id: str, v: int, artifact: dict[str, Any], card: dict[str, Any], input_schema: dict[str, Any]) -> Path:
    d = version_dir(root, model_id, v)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.joblib"
    joblib.dump(artifact, path, compress=3)
    (d / "card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    (d / "input_schema.json").write_text(json.dumps(input_schema, indent=2, default=str), encoding="utf-8")
    _cache.pop(str(path), None)
    return path


_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_lock = threading.Lock()


def load_artifact(path: str | Path) -> dict[str, Any]:
    key = str(path)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    art: dict[str, Any] = joblib.load(path)
    with _lock:
        _cache[key] = art
        while len(_cache) > 6:
            _cache.popitem(last=False)
    return art


class Tracker:
    """Best-effort MLflow logging: tracking problems never break training."""

    def __init__(self, enabled: bool, uri: str, experiment: str) -> None:
        self.enabled = enabled
        self.experiment = experiment
        self._mlflow: Any = None
        if enabled:
            try:
                os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                import mlflow

                if uri.startswith("sqlite:///"):
                    Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(experiment)
                self._mlflow = mlflow
            except Exception as e:
                log.warning("MLflow disabled: %s", e.__class__.__name__)
                self.enabled = False

    def log_run(self, name: str, params: dict[str, Any], metrics: dict[str, float], tags: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        try:
            with self._mlflow.start_run(run_name=name):
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self._mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and v == v})
                if tags:
                    self._mlflow.set_tags(tags)
        except Exception as e:
            log.warning("MLflow logging failed: %s", e.__class__.__name__)

    @contextmanager
    def quiet(self) -> Iterator[None]:
        yield
