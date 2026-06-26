from __future__ import annotations

import sys
from pathlib import Path

# Make this runnable from ANY cwd / launch form:
#   uvicorn main:app                        (from this folder)
#   uvicorn ver2.app.backend.main:app       (from the repo root)
# - this folder on sys.path  -> sibling imports (inference_service, schemas)
# - repo root on sys.path    -> the `ver2.*` model code inference_service pulls in
_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference_service import InferenceService
from schemas import HealthResponse, PredictRequest, PredictResponse


APP_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_DIR / "frontend"

app = FastAPI(title="DeDe Inference App", version="1.0")
service = InferenceService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, model_loaded=service.model_loaded)


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = service.predict(
            url=payload.url,
            title=payload.title,
            body=payload.body,
            upvotes=payload.upvotes,
            num_comments=payload.num_comments,
            created_utc=payload.created_utc,
            translate=payload.translate,
        )
        return PredictResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
