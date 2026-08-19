from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ragagent import __version__
from ragagent.api import router
from ragagent.config import Settings, get_settings
from ragagent.container import Container


def create_app(settings: Settings | None = None) -> FastAPI:
    selected_settings = settings or get_settings()
    container = Container(selected_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        container.initialize()
        yield

    app = FastAPI(
        title="Traceable Agentic RAG",
        version=__version__,
        summary="Standalone local RAG Agent application and embeddable REST runtime",
        lifespan=lifespan,
    )
    app.state.container = container
    app.include_router(router)

    @app.middleware("http")
    async def prevent_stale_frontend(request, call_next):
        response = await call_next(request)
        if request.url.path == "/app" or request.url.path.startswith("/app/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    static_dir = Path(__file__).parent / "static"
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="app")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/")

    return app


app = create_app()
