from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent_service.api.chat import router as chat_router
from agent_service.api.health import router as health_router
from agent_service.core.config import get_settings
from agent_service.core.errors import register_exception_handlers
from agent_service.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="HYEEE Agent Service",
        version=settings.service_version,
        description="Python Agent service for HYEEE AI chat.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/v1/agent")
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/ui/chat.html")

    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "agent_service.main:app",
        host=settings.agent_host,
        port=settings.agent_port,
        reload=settings.agent_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
