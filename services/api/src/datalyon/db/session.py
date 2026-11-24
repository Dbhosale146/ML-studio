from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import API_ROOT


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        connect_args: dict[str, Any] = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn: Any, _: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")  # readers (SSE) do not block the training writer
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def migrate(self) -> None:
        if self.url.startswith("sqlite:///"):
            Path(self.url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        cfg = Config(str(API_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(API_ROOT / "alembic"))
        cfg.set_main_option("sqlalchemy.url", self.url)
        cfg.attributes["configure_logger"] = False
        command.upgrade(cfg, "head")
