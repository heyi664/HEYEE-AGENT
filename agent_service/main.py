from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent_service.api.chat import router as chat_router
from agent_service.api.health import router as health_router
from agent_service.api.knowledge import router as knowledge_router
from agent_service.core.config import get_settings
from agent_service.core.errors import register_exception_handlers
from agent_service.core.logging import configure_logging
from agent_service.mcp.adapter import register_mcp_tools
from agent_service.mcp.http_client import StreamableHttpMcpClient
from agent_service.middleware.upload_rate_limit import UploadRateLimitMiddleware
from agent_service.tools.builtin import register_builtin_tools
from agent_service.tools.registry import tool_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    mcp_client: StreamableHttpMcpClient | None = None
    register_builtin_tools(tool_registry)

    if settings.mcp_enabled:
        mcp_client = StreamableHttpMcpClient(
            server_url=settings.mcp_server_url,
            token=settings.mcp_server_token,
            timeout_seconds=settings.mcp_timeout_seconds,
        )
        try:
            server_info = await mcp_client.initialize()
            registered = await register_mcp_tools(
                mcp_client,
                tool_registry,
                name_prefix=settings.mcp_tool_prefix,
            )
            app.state.mcp_client = mcp_client
            app.state.mcp_tools = registered
            logger.info(
                "Java MCP connected server=%s tools=%s",
                server_info.get("serverInfo", {}).get("name", "unknown"),
                registered,
            )
        except Exception:
            await mcp_client.close()
            mcp_client = None
            logger.exception("Java MCP initialization failed url=%s", settings.mcp_server_url)
            if settings.mcp_fail_fast:
                raise

    try:
        yield
    finally:
        if mcp_client is not None:
            await mcp_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="HYEEE Agent Service",
        version=settings.service_version,
        description="Python Agent service for HYEEE AI chat.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(UploadRateLimitMiddleware, settings=settings)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/v1/agent")
    app.include_router(knowledge_router, prefix="/v1")
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
