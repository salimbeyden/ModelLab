"""
ModelLab API: Main application entry point.
Clean, modular FastAPI application with proper lifecycle management.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.model_factory import model_factory

# Import plugins to trigger registration
import app.plugins.models  # noqa: F401

# Ensure required directories exist (important for Railway volumes)
os.makedirs("data", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("tmp_r", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Startup: Initialize resources, ensure directories exist.
    Shutdown: Cleanup resources if needed.
    """
    # Startup
    print(f"[ModelLab] Starting up...")
    print(f"[ModelLab] Data directory: {settings.DATA_DIR}")
    print(f"[ModelLab] Runs directory: {settings.RUNS_DIR}")
    
    yield
    
    # Shutdown
    print("[ModelLab] Shutting down...")


def create_app() -> FastAPI:
    """
    Application factory pattern for creating the FastAPI app.
    Enables easy testing and configuration.
    """
    app = FastAPI(
        title=settings.API_TITLE,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # Register routes
    _register_routes(app)
    
    # Mount static files
    _mount_static_files(app)
    
    return app


def _register_routes(app: FastAPI):
    """Register all API routers."""
    from app.api import routes, explanation_routes
    
    # Health check
    @app.get("/", tags=["Health"])
    def health_check():
        return {"status": "ok", "service": "ModelLab Backend"}
    
    # Plugin discovery
    @app.get("/v1/plugins/models", tags=["Plugins"])
    def list_available_models():
        return model_factory.list_available_models()
    
    # API routers
    app.include_router(routes.router, prefix="/v1")
    app.include_router(explanation_routes.router, prefix="/v1")


def _mount_static_files(app: FastAPI):
    """Mount static file directories."""
    # Serve run artifacts directly (MVP approach)
    # In production, use Nginx or signed URLs
    app.mount(
        "/runs",
        StaticFiles(directory=str(settings.RUNS_DIR)),
        name="runs"
    )


# Create the application instance
app = create_app()


