from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model provider cannot produce a response."""


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("request validation failed path=%s errors=%s", request.url.path, exc.errors())
        return JSONResponse(status_code=400, content={"detail": "Invalid chat request"})

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        logger.warning("model unavailable path=%s error=%s", request.url.path, exc)
        return JSONResponse(status_code=503, content={"detail": "AI model is unavailable"})

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.info(
            "http error path=%s status=%s detail=%s",
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected error path=%s", request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Agent internal error"})
