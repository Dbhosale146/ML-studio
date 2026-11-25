"""Playground: single predictions with drivers, batch scoring of an uploaded sheet, forecasts. Internal to the app."""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...core.errors import AppError, NotFoundError
from ...core.ids import new_id
from ...db.models import PredictionJob
from ...db.repositories.activity import add_activity
from ...db.repositories.models import ModelRepo
from ...db.session import Database
from ...ml.inference import batch_frame, forecast_view, predict_tabular, validate_rows
from ...ml.ingest import IngestError, read_file, sniff_file_type
from ...ml.registry import load_artifact
from ...schemas.api import BatchJobOut, ErrorEnvelope, PredictIn, PredictOut
from ...storage.file_store import FileTooLargeError

router = APIRouter(tags=["predictions"], responses={404: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}})
log = logging.getLogger("datalyon.predictions")


def _artifact(request: Request, model_id: str) -> tuple[Any, dict[str, Any], Any]:
    repo = ModelRepo(request.app.state.db)
    m = repo.get(model_id)
    if m.status != "ready" or not m.latest_version:
        raise AppError("model_not_ready", "This model has no published version yet.", 409)
    v = repo.version(model_id, m.latest_version)
    return m, load_artifact(v.artifact_path), v


@router.post(
    "/models/{model_id}/predict", response_model=PredictOut, summary="Predict one or more rows (or a forecast) with per-row drivers"
)
def predict(model_id: str, body: PredictIn, request: Request) -> dict[str, Any]:
    m, art, v = _artifact(request, model_id)
    repo = ModelRepo(request.app.state.db)
    if art["kind"] == "forecast":
        view = forecast_view(art, body.series, body.horizon)
        repo.increment_predictions(model_id, len(view["table"]))
        return {"model_id": model_id, "version": v.version, "problem_type": m.problem_type, "rows": [], "forecast": view}
    if not body.rows:
        raise AppError("validation_error", "Send at least one row to score.", 422)
    X, warnings = validate_rows(body.rows, art["input_schema"])
    results, explainer = predict_tabular(art, X, explain=body.explain)
    for r, w in zip(results, warnings, strict=True):
        r["warnings"] = w
    repo.increment_predictions(model_id, len(results))
    return {"model_id": model_id, "version": v.version, "problem_type": m.problem_type, "rows": results, "explainer": explainer}


def _job_out(job: PredictionJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "model_id": job.model_id,
        "status": job.status,
        "file_name": job.file_name,
        "rows": job.rows,
        "summary": job.summary,
        "error": job.error,
        "created_at": job.created_at,
        "download_url": f"/api/v1/predictions/{job.id}/download" if job.status == "done" else None,
    }


def _run_batch(
    db: Database, job_id: str, art: dict[str, Any], path: str, file_type: str, out_path: str, max_rows: int, max_cols: int, model_name: str
) -> None:
    def update(**f: Any) -> None:
        with db.session() as s:
            j = s.get(PredictionJob, job_id)
            if j is not None:
                for k, v in f.items():
                    setattr(j, k, v)

    update(status="running")
    try:
        sheets = read_file(path, file_type, max_rows, max_cols)  # type: ignore[arg-type]
        cols = set(art["columns"])
        sheet = max(sheets, key=lambda s: (len(cols & set(map(str, s.df.columns))), len(s.df)))
        out, report = batch_frame(art, sheet.df)
        with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
            out.to_excel(xw, sheet_name="Predictions", index=False)
        preview = out.head(20).astype(object).where(out.head(20).notna(), None)
        report["sheet"] = sheet.name
        report["preview"] = {
            "columns": [str(c) for c in out.columns],
            "rows": [[None if v is None else str(v)[:40] for v in r] for r in preview.itertuples(index=False, name=None)],
        }
        update(status="done", rows=len(out), summary=report)
        with db.session() as s:
            j = s.get(PredictionJob, job_id)
            if j is not None:
                ModelRepo(db).increment_predictions(j.model_id, len(out))
                add_activity(db, "predict", f"Scored **{len(out):,} rows** of {j.file_name} with **{model_name}**", model_id=j.model_id)
    except (IngestError, ValueError) as e:
        update(status="failed", error=getattr(e, "message", None) or str(e))
    except Exception:
        log.exception("batch prediction failed")
        update(status="failed", error="The batch prediction failed. The details are in the server log.")


@router.post(
    "/models/{model_id}/predict-batch",
    response_model=BatchJobOut,
    status_code=202,
    summary="Score an uploaded Excel/CSV file in the background",
)
def predict_batch(model_id: str, request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    m, art, v = _artifact(request, model_id)
    if art["kind"] == "forecast":
        raise AppError("not_supported", "Forecasting models do not score uploaded rows; download the forecast instead.", 409)
    settings, store = request.app.state.settings, request.app.state.store
    jid = new_id("pj")
    head = file.file.read(8192)
    file.file.seek(0)
    try:
        ftype = sniff_file_type(file.filename or "", head)
    except IngestError as e:
        raise AppError(e.code, e.message, 415) from e
    try:
        path, _, _ = store.save_stream(
            "uploads", f"batch/{jid}/{(file.filename or 'upload').replace('/', '_')}", file.file, settings.max_upload_mb * 1024 * 1024
        )
    except FileTooLargeError as e:
        raise AppError("file_too_large", f"The file is larger than the {settings.max_upload_mb} MB limit.", 413) from e
    out_path = store.path("outputs", f"{jid}.xlsx")
    job = PredictionJob(
        id=jid,
        model_id=model_id,
        version=v.version,
        status="queued",
        file_name=file.filename or "upload",
        input_path=str(path),
        output_path=str(out_path),
    )
    db: Database = request.app.state.db
    with db.session() as s:
        s.add(job)
    request.app.state.runner.submit(
        f"batch-{jid}", lambda: _run_batch(db, jid, art, str(path), ftype, str(out_path), settings.max_rows, settings.max_columns, m.name)
    )
    return _job_out(job)


@router.get("/predictions/{job_id}", response_model=BatchJobOut, summary="Status and summary of a batch prediction")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    with request.app.state.db.session() as s:
        job = s.get(PredictionJob, job_id)
        if job is None:
            raise NotFoundError("Prediction job", job_id)
        return _job_out(job)


@router.get(
    "/predictions/{job_id}/download", summary="Download the scored file (.xlsx with prediction columns)", response_class=FileResponse
)
def download(job_id: str, request: Request) -> FileResponse:
    with request.app.state.db.session() as s:
        job = s.get(PredictionJob, job_id)
        if job is None:
            raise NotFoundError("Prediction job", job_id)
        if job.status != "done" or not job.output_path:
            raise AppError("not_ready", "The predictions are not ready yet.", 409)
        stem = job.file_name.rsplit(".", 1)[0]
        return FileResponse(
            job.output_path,
            filename=f"{stem}_predictions.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@router.get("/models/{model_id}/forecast/download", summary="Download a forecast as .xlsx", response_class=StreamingResponse)
def download_forecast(
    model_id: str, request: Request, series: str | None = Query(None), horizon: int | None = Query(None, ge=1, le=1000)
) -> StreamingResponse:
    m, art, _ = _artifact(request, model_id)
    if art["kind"] != "forecast":
        raise AppError("not_supported", "Only forecasting models have a forecast to download.", 409)
    view = forecast_view(art, series, horizon)
    buf = io.BytesIO()
    pd.DataFrame(view["table"]).assign(series=view["series"]).to_excel(buf, index=False, sheet_name="Forecast")
    buf.seek(0)
    name = f"{m.slug}_{view['series']}_forecast.xlsx".replace(" ", "_")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
