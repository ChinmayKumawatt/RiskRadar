from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BulkPredictionResponse,
    MetadataResponse,
    PredictionResponse,
    RiskInput,
    prediction_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = prediction_service
    yield


app = FastAPI(
    title="RiskRadar API",
    version="1.0.0",
    description=(
        "FastAPI wrapper for multi-disease inference. "
        "Shared inputs such as age can be sent once and reused across eligible models."
    ),
    lifespan=lifespan,
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_prediction_service():
    return app.state.prediction_service


@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")


@app.get("/api")
def api_root():
    return {
        "message": "RiskRadar API is running",
        "docs": "/docs",
        "metadata": "/metadata",
    }


@app.get("/health")
def health_check():
    service = get_prediction_service()
    return {
        "status": "ok",
        "supported_diseases": sorted(service.predictors),
    }


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    service = get_prediction_service()
    return service.get_metadata()


@app.post("/predict/all", response_model=BulkPredictionResponse)
def predict_all(payload: RiskInput):
    service = get_prediction_service()
    return service.predict_all(payload.model_dump())


@app.post("/predict/{disease_name}", response_model=PredictionResponse)
def predict_disease(disease_name: str, payload: RiskInput):
    service = get_prediction_service()

    try:
        return service.predict_for_disease(
            disease_name=disease_name,
            request_payload=payload.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
