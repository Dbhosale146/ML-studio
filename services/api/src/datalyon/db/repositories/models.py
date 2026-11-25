from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from ...core.errors import NotFoundError
from ...core.ids import new_id
from ..models import Dataset, Model, ModelVersion
from ..session import Database


class ModelRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, **fields: Any) -> Model:
        m = Model(id=new_id("mdl"), **fields)
        with self.db.session() as s:
            s.add(m)
        return m

    def update(self, mid: str, **fields: Any) -> Model:
        with self.db.session() as s:
            m = s.get(Model, mid)
            if m is None:
                raise NotFoundError("Model", mid)
            for k, v in fields.items():
                setattr(m, k, v)
            return m

    def get(self, mid: str) -> Model:
        with self.db.session() as s:
            m = s.get(Model, mid)
            if m is None:
                raise NotFoundError("Model", mid)
            return m

    def list_page(
        self, limit: int = 50, offset: int = 0, status: str | None = None, problem_type: str | None = None, q: str | None = None
    ) -> tuple[list[Model], int]:
        stmt = select(Model)
        if status:
            stmt = stmt.where(Model.status == status)
        if problem_type:
            stmt = stmt.where(Model.problem_type == problem_type)
        if q:
            like = f"%{q.lower()}%"
            stmt = stmt.where(
                func.lower(Model.name).like(like)
                | func.lower(Model.objective).like(like)
                | func.lower(func.coalesce(Model.target, "")).like(like)
            )
        with self.db.session() as s:
            total = int(s.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
            items = list(s.scalars(stmt.order_by(Model.updated_at.desc()).limit(limit).offset(offset)))
            return items, total

    def add_version(
        self, mid: str, version: int, artifact_path: str, card: dict[str, Any], input_schema: dict[str, Any], data_hash: str
    ) -> ModelVersion:
        v = ModelVersion(
            model_id=mid, version=version, artifact_path=artifact_path, card=card, input_schema=input_schema, data_hash=data_hash
        )
        with self.db.session() as s:
            s.add(v)
        return v

    def versions(self, mid: str) -> list[ModelVersion]:
        with self.db.session() as s:
            return list(s.scalars(select(ModelVersion).where(ModelVersion.model_id == mid).order_by(ModelVersion.version.desc())))

    def version(self, mid: str, version: int | None = None) -> ModelVersion:
        with self.db.session() as s:
            stmt = select(ModelVersion).where(ModelVersion.model_id == mid)
            stmt = stmt.where(ModelVersion.version == version) if version else stmt.order_by(ModelVersion.version.desc())
            v = s.scalars(stmt.limit(1)).first()
            if v is None:
                raise NotFoundError("Model version", f"{mid} v{version or 'latest'}")
            return v

    def next_version(self, mid: str) -> int:
        with self.db.session() as s:
            return int(s.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.model_id == mid)) or 0) + 1

    def delete(self, mid: str) -> None:
        with self.db.session() as s:
            m = s.get(Model, mid)
            if m is None:
                raise NotFoundError("Model", mid)
            s.delete(m)

    def increment_predictions(self, mid: str, n: int) -> None:
        with self.db.session() as s:
            m = s.get(Model, mid)
            if m is not None:
                m.prediction_count = (m.prediction_count or 0) + n

    def dataset_names(self, ids: list[str]) -> dict[str, str]:
        ids = [i for i in ids if i]
        if not ids:
            return {}
        with self.db.session() as s:
            return {d.id: d.file_name for d in s.scalars(select(Dataset).where(Dataset.id.in_(ids)))}
