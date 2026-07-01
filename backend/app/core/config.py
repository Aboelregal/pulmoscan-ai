"""
Central configuration for PulmoScan AI backend.
All tunables live here so modules never hardcode paths/secrets.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "PulmoScan AI"
    API_V1_PREFIX: str = "/api/v1"

    # Storage
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    CACHE_DIR: Path = DATA_DIR / "cache"
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    REPORTS_DIR: Path = DATA_DIR / "reports"

    # Database
    DATABASE_URL: str = "postgresql://ctvision:ctvision@db:5432/ctvision"

    # Auth
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8

    # Model weights (drop trained checkpoints here; see README)
    LUNG_SEG_WEIGHTS: str = "lung_segmentation_segresnet.pt"
    NODULE_DET_WEIGHTS: str = "nodule_detector_nnunet.pt"
    NODULE_SEG_WEIGHTS: str = "nodule_segmentation_swin_unetr.pt"
    CLASSIFIER_WEIGHTS: str = "disease_classifier_densenet3d.pt"

    # LLM
    LLM_MODEL_NAME: str = "google/medgemma-4b-it"  # falls back to template mode if unavailable
    LLM_USE_REMOTE_API: bool = False
    ANTHROPIC_API_KEY: str | None = None

    # RAG
    RAG_INDEX_PATH: str = "rag_faiss.index"
    RAG_EMBED_MODEL: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

    # Device
    DEVICE: str = "cpu"  # set to "cuda" if GPU available

    class Config:
        env_file = ".env"


settings = Settings()

for d in (settings.UPLOAD_DIR, settings.CACHE_DIR, settings.WEIGHTS_DIR, settings.REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
