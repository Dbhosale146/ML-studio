from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from ...db.models import Model
from ...db.repositories.activity import add_activity
from ...db.repositories.models import ModelRepo
from ...schemas.api import ErrorEnvelope, InputSchemaOut, ModelDetailOut, ModelSummaryOut, ModelVersionOut, Page

router = APIRouter(prefix="/models", tags=["models"], responses={404: {"model": ErrorEnvelope}})


def summary(m: Model, dataset_names: dict[str, str]) -> dict[str, Any]:
    return {
        "id": m.id,
        "name": m.name,
        "slug": m.slug,
        "problem_type": m.problem_type,
        "problem_label": m.problem_label,
        "objective": m.objective,
        "status": m.status,
        "progress": m.progress,
        "stage": m.stage,
        "target": m.target,
        "algorithm": m.algorithm,
        "latest_version": m.latest_version,
        "primary_metric": m.primary_metric,
        "baseline_metric": m.baseline_metric,
        "dataset_id": m.dataset_id,
        "dataset_name": dataset_names.get(m.dataset_id or ""),
        "session_id": m.session_id,
        "prediction_count": m.prediction_count,
        "error": m.error,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
    }


@router.get("", response_model=Page[ModelSummaryOut], summary="List models (search, filter by problem type and status)")
def list_models(
    request: Request,
    q: str | None = Query(None, max_length=200),
    status: str | None = Query(None, pattern="^(draft|training|ready|failed)$"),
    problem_type: str | None = Query(None, pattern="^(binary|multiclass|regression|forecasting)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    repo = ModelRepo(request.app.state.db)
    items, total = repo.list_page(limit, offset, status, problem_type, q)
    names = repo.dataset_names([m.dataset_id or "" for m in items])
    return {"items": [summary(m, names) for m in items], "total": total, "limit": limit, "offset": offset}


@router.get("/{model_id}", response_model=ModelDetailOut, summary="Model with its latest model card")
def get_model(model_id: str, request: Request) -> dict[str, Any]:
    repo = ModelRepo(request.app.state.db)
    m = repo.get(model_id)
    versions = repo.versions(model_id)
    out = summary(m, repo.dataset_names([m.dataset_id or ""]))
    out["card"] = versions[0].card if versions else None
    out["versions"] = [_version(v) for v in versions]
    return out


def _version(v: Any) -> dict[str, Any]:
    card = v.card or {}
    primary = next(iter(card.get("metrics") or []), None)
    return {
        "version": v.version,
        "created_at": v.created_at,
        "data_hash": v.data_hash,
        "primary_metric": {"name": primary["name"], "display": primary["display"], "value": primary.get("value")} if primary else None,
        "algorithm": card.get("algorithm"),
        "note": f"Trained on {card.get('dataset', {}).get('file_name', '')} ({card.get('dataset', {}).get('training_rows', '?')} rows)",
    }


@router.get("/{model_id}/versions", response_model=list[ModelVersionOut])
def list_versions(model_id: str, request: Request) -> list[dict[str, Any]]:
    repo = ModelRepo(request.app.state.db)
    repo.get(model_id)
    return [_version(v) for v in repo.versions(model_id)]


@router.get(
    "/{model_id}/input-schema",
    response_model=InputSchemaOut,
    summary="Fields the Test-model form needs (types, categories, ranges, defaults)",
)
def input_schema(model_id: str, request: Request, version: int | None = Query(None, ge=1)) -> dict[str, Any]:
    repo = ModelRepo(request.app.state.db)
    m = repo.get(model_id)
    v = repo.version(model_id, version)
    doc = v.input_schema or {}
    return {
        "model_id": model_id,
        "version": v.version,
        "problem_type": m.problem_type,
        "target": m.target,
        "fields": doc.get("fields", []),
        "examples": doc.get("examples", []),
        "forecast": doc.get("forecast"),
    }


@router.delete("/{model_id}", status_code=204, summary="Delete a model, its versions and artifacts")
def delete_model(model_id: str, request: Request) -> None:
    repo = ModelRepo(request.app.state.db)
    m = repo.get(model_id)
    repo.delete(model_id)
    request.app.state.store.delete("artifacts", f"models/{model_id}")
    add_activity(request.app.state.db, "model", f"Deleted model **{m.name}**", model_id=model_id)
