from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import init_db
from app.api import patient_routes, study_routes, auth_routes

app = FastAPI(
    title=settings.APP_NAME,
    description="AI assistant for Chest CT analysis: segmentation, detection, "
                "classification, explainability, and RAG-grounded reporting.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(patient_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(study_routes.router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": settings.APP_NAME}
