"""FastAPI app factory: config, logging, DB migrations, storage, routers and error handlers."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import health
from .core.config import Settings, get_settings
from .core.errors import install_error_handlers
from .core.logging import request_id_var, setup_logging
from .db.session import Database
from .storage.file_store import LocalFileStore

log = logging.getLogger("datalyon.http")

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.storage.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_url)
    store = LocalFileStore(settings.storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.migrate()
        startup = getattr(app.state, "on_startup", None)
        if startup:
            startup()
        log.info("Datalyon API ready", extra={"path": settings.db_url})
        yield
        shutdown = getattr(app.state, "on_shutdown", None)
        if shutdown:
            shutdown()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Internal API of the Datalyon web app (not a public prediction API).",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.store = store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        if not request.url.path.endswith("/events"):
            log.info(
                "request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
        return response

    install_error_handlers(app)
    app.include_router(health.router, prefix=API_PREFIX)
    _include_feature_routers(app)
    return app


def _include_feature_routers(app: FastAPI) -> None:
    """Routers added milestone by milestone; imported lazily so the skeleton runs on its own."""
    import importlib

    for name in ("datasets", "sessions", "models", "predictions", "activity"):
        try:
            mod = importlib.import_module(f"datalyon.api.routes.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"datalyon.api.routes.{name}":
                continue
            raise
        app.include_router(mod.router, prefix=API_PREFIX)
    try:
        from .agent.orchestrator import attach_agent

        attach_agent(app)
    except ModuleNotFoundError as e:
        if e.name != "datalyon.agent.orchestrator":
            raise


app = create_app()
