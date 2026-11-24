from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

from fastapi import Request
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..db.session import Database
from ..storage.file_store import LocalFileStore

if TYPE_CHECKING:
    from ..agent.orchestrator import AgentService


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.db)


def get_db(request: Request) -> Iterator[Session]:
    db: Database = request.app.state.db
    with db.session() as s:
        yield s


def get_store(request: Request) -> LocalFileStore:
    return cast(LocalFileStore, request.app.state.store)


def get_agent(request: Request) -> AgentService:
    return cast("AgentService", request.app.state.agent)
