from __future__ import annotations

from sqlalchemy import select

from ..models import Activity
from ..session import Database


def add_activity(
    db: Database, type_: str, text: str, *, model_id: str | None = None, dataset_id: str | None = None, session_id: str | None = None
) -> None:
    with db.session() as s:
        s.add(Activity(type=type_, text=text, model_id=model_id, dataset_id=dataset_id, session_id=session_id))


def list_activity(db: Database, limit: int = 30) -> list[Activity]:
    with db.session() as s:
        return list(s.scalars(select(Activity).order_by(Activity.id.desc()).limit(limit)))
