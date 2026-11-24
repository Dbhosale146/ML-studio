from typing import Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ...db.repositories.datasets import DatasetRepo
from ...schemas.api import DatasetOut, ErrorEnvelope, Page, PreviewOut, SheetChoiceIn

router = APIRouter(prefix="/datasets", tags=["datasets"], responses={404: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}})


def repo(request: Request) -> DatasetRepo:
    return DatasetRepo(request.app.state.db, request.app.state.store, request.app.state.settings)


@router.post(
    "",
    response_model=DatasetOut,
    status_code=201,
    summary="Upload an Excel or CSV file",
    responses={413: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}},
)
def upload_dataset(file: UploadFile = File(..., description=".xlsx, .xls or .csv"), r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    ds = r.create_from_upload(file.filename or "upload", file.file)
    return r.to_out(ds)


@router.post("/{dataset_id}/sheet", response_model=DatasetOut, summary="Choose the sheet (and optionally the header row) to use")
def choose_sheet(dataset_id: str, body: SheetChoiceIn, r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    return r.to_out(r.select_sheet(dataset_id, body.sheet, body.header_row))


@router.get("", response_model=Page[DatasetOut], summary="List datasets")
def list_datasets(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    items, total = r.list_page(limit, offset)
    return {"items": [r.to_out(d) for d in items], "total": total, "limit": limit, "offset": offset}


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    return r.to_out(r.get(dataset_id))


@router.get("/{dataset_id}/preview", response_model=PreviewOut, summary="First rows of the chosen sheet")
def preview_dataset(dataset_id: str, limit: int = Query(50, ge=1, le=500), r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    return r.preview(dataset_id, limit)


@router.get("/{dataset_id}/profile", summary="Column profile and data-quality report")
def profile_dataset(dataset_id: str, r: DatasetRepo = Depends(repo)) -> dict[str, Any]:
    ds = r.get(dataset_id)
    if ds.profile is None:
        r.load_df(dataset_id)  # raises a clear 409 when no sheet is chosen yet
    return ds.profile or {}


@router.delete("/{dataset_id}", status_code=204, summary="Delete a dataset and its files")
def delete_dataset(dataset_id: str, r: DatasetRepo = Depends(repo)) -> None:
    r.delete(dataset_id)
