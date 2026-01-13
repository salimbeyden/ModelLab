from fastapi import FastAPI
from app.core.model_factory import model_factory
import app.plugins.models # This triggers registration

app = FastAPI(title="ModelLab API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "ModelLab Backend"}

@app.get("/v1/plugins/models")
def get_models():
    return model_factory.list_available_models()

from fastapi.staticfiles import StaticFiles

from app.api import routes
from app.api import explanation_routes

app.include_router(routes.router, prefix="/v1")
app.include_router(explanation_routes.router, prefix="/v1")

# Mount runs directory to serve artifacts directly (MVP)
# In production, use Nginx or signed URLs
app.mount("/runs", StaticFiles(directory="runs"), name="runs")


