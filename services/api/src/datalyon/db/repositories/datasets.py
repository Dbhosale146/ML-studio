"""Dataset lifecycle: upload -> parse every sheet -> choose a sheet -> profile. Used by the routes and the agent."""

from __future__ import annotations

import logging
import re
from typing import Any, BinaryIO, cast

import pandas as pd
from sqlalchemy import func, select

from ...core.config import Settings
from ...core.errors import AppError, NotFoundError
from ...core.ids import new_id
from ...ml.ingest import FileType, IngestError, dataset_hash, read_file, reparse_sheet, sniff_file_type
from ...ml.profiling import preview, profile_dataframe
from ...storage.file_store import FileTooLargeError, LocalFileStore
from ..models import Dataset, Model
from ..session import Database
from .activity import add_activity

log = logging.getLogger("datalyon.datasets")


def _safe_name(name: str) -> str:
    base = re.sub(r"[^\w.\- ]+", "_", name or "upload").strip() or "upload"
    return base[-120:]


class DatasetRepo:
    def __init__(self, db: Database, store: LocalFileStore, settings: Settings) -> None:
        self.db, self.store, self.settings = db, store, settings

    # ---- upload & sheet selection -------------------------------------------------------
    def create_from_upload(self, filename: str, stream: BinaryIO, session_id: str | None = None) -> Dataset:
        did = new_id("ds")
        head = stream.read(8192)
        stream.seek(0)
        try:
            ftype: FileType = sniff_file_type(filename, head)
        except IngestError as e:
            raise AppError(e.code, e.message, 415) from e
        try:
            path, size, sha = self.store.save_stream(
                "uploads", f"{did}/{_safe_name(filename)}", stream, self.settings.max_upload_mb * 1024 * 1024
            )
        except FileTooLargeError as e:
            raise AppError("file_too_large", f"The file is larger than the {self.settings.max_upload_mb} MB limit.", 413) from e
        try:
            sheets = read_file(path, ftype, self.settings.max_rows, self.settings.max_columns)
        except IngestError as e:
            self.store.delete("uploads", did)
            raise AppError(e.code, e.message, 422) from e
        infos: list[dict[str, Any]] = []
        for i, sf in enumerate(sheets):
            cache = self.store.path("datasets", f"{did}/sheet_{i}.pkl")
            sf.df.to_pickle(cache)
            infos.append(
                {
                    "name": sf.name,
                    "rows": len(sf.df),
                    "cols": sf.df.shape[1],
                    "header_row": sf.header_row,
                    "notes": sf.notes,
                    "cache": cache.name,
                }
            )
        name = re.sub(r"\.(xlsx|xlsm|xls|csv|txt)$", "", filename or "dataset", flags=re.I)
        ds = Dataset(
            id=did,
            name=name[:255],
            file_name=_safe_name(filename),
            file_type=ftype,
            size_bytes=size,
            file_hash=sha,
            file_path=str(path),
            sheets=infos,
            status="uploaded",
        )
        with self.db.session() as s:
            s.add(ds)
        add_activity(
            self.db,
            "data",
            f"Uploaded **{ds.file_name}** ({len(infos)} sheet{'s' if len(infos) > 1 else ''})",
            dataset_id=did,
            session_id=session_id,
        )
        log.info("dataset uploaded", extra={"dataset_id": did})
        if len(infos) == 1:
            return self.select_sheet(did, infos[0]["name"])
        return ds

    def select_sheet(self, did: str, sheet: str, header_row: int | None = None) -> Dataset:
        ds = self.get(did)
        info = next((x for x in ds.sheets if x["name"] == sheet), None)
        if info is None:
            raise AppError("unknown_sheet", f"Sheet '{sheet}' does not exist in {ds.file_name}.", 404)
        if header_row is not None and header_row != info.get("header_row"):
            try:
                sf = reparse_sheet(
                    self.store.path("uploads", f"{did}/{ds.file_name}"),
                    cast(FileType, ds.file_type),
                    sheet,
                    header_row,
                    self.settings.max_rows,
                    self.settings.max_columns,
                )
            except IngestError as e:
                raise AppError(e.code, e.message, 422) from e
            df, notes, hr = sf.df, sf.notes, sf.header_row
        else:
            df = pd.read_pickle(self.store.path("datasets", f"{did}/{info['cache']}"))
            notes, hr = info.get("notes", []), info.get("header_row")
        data_path = self.store.path("datasets", f"{did}/data.pkl")
        df.to_pickle(data_path)
        profile = profile_dataframe(df, ds.file_name, sheet, notes)
        with self.db.session() as s:
            row = s.get(Dataset, did)
            assert row is not None
            row.sheet, row.header_row, row.rows, row.cols = sheet, hr, len(df), int(df.shape[1])
            row.data_path, row.profile, row.data_hash, row.status = str(data_path), profile, dataset_hash(df), "ready"
            ds = row
        return ds

    # ---- reads --------------------------------------------------------------------------
    def get(self, did: str) -> Dataset:
        with self.db.session() as s:
            ds = s.get(Dataset, did)
            if ds is None:
                raise NotFoundError("Dataset", did)
            return ds

    def load_df(self, did: str) -> pd.DataFrame:
        ds = self.get(did)
        if ds.status != "ready" or not ds.data_path:
            raise AppError("sheet_not_selected", "Choose a sheet for this dataset first.", 409)
        return pd.read_pickle(ds.data_path)

    def list_page(self, limit: int = 50, offset: int = 0) -> tuple[list[Dataset], int]:
        with self.db.session() as s:
            total = int(s.scalar(select(func.count()).select_from(Dataset)) or 0)
            items = list(s.scalars(select(Dataset).order_by(Dataset.created_at.desc()).limit(limit).offset(offset)))
            return items, total

    def preview(self, did: str, limit: int) -> dict[str, Any]:
        return preview(self.load_df(did), limit)

    def used_by(self, did: str) -> list[str]:
        with self.db.session() as s:
            return list(s.scalars(select(Model.id).where(Model.dataset_id == did)))

    def delete(self, did: str) -> None:
        ds = self.get(did)
        with self.db.session() as s:
            row = s.get(Dataset, did)
            if row is not None:
                s.delete(row)
        self.store.delete("uploads", did)
        self.store.delete("datasets", did)
        add_activity(self.db, "data", f"Deleted dataset **{ds.file_name}**", dataset_id=did)

    def to_out(self, ds: Dataset) -> dict[str, Any]:
        return {
            "id": ds.id,
            "name": ds.name,
            "file_name": ds.file_name,
            "file_type": ds.file_type,
            "size_bytes": ds.size_bytes,
            "status": ds.status,
            "sheets": ds.sheets,
            "sheet": ds.sheet,
            "header_row": ds.header_row,
            "rows": ds.rows,
            "cols": ds.cols,
            "quality": (ds.profile or {}).get("quality"),
            "needs_sheet": ds.status != "ready",
            "used_by": self.used_by(ds.id),
            "created_at": ds.created_at,
        }
