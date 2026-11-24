"""Consistent error envelope: {"error": {"code", "message", "details"}, "request_id"}."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging import request_id_var

log = logging.getLogger("datalyon.errors")


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


class NotFoundError(AppError):
    def __init__(self, what: str, id_: str) -> None:
        super().__init__("not_found", f"{what} '{id_}' was not found.", 404)


def _envelope(code: str, message: str, details: Any, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details}, "request_id": request_id_var.get()},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _envelope(exc.code, exc.message, exc.details, exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return _envelope("validation_error", "The request is not valid.", details, 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        return _envelope(code, str(exc.detail), None, exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return _envelope("internal_error", "Something went wrong on our side. The details are in the server log.", None, 500)
