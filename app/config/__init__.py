"""Application configuration and environment management.

Privacy & Security:
- Tracing disabled: LANGCHAIN_TRACING_V2=false, LANGSMITH_TRACING=false.
- Strict environment validation at startup.
- Safe logging only (no notes, no keys).
"""

import os
import sys
from typing import Set, Optional
from dotenv import load_dotenv

# Disable all third-party tracing immediately to guarantee privacy
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Load environment variables from .env (explicit backend path fallback)
_backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_env_file = os.path.join(_backend_dir, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file, override=True)
load_dotenv(override=True)

# Environment settings
APP_ENV: str = os.environ.get("APP_ENV", "development")
SUPABASE_JWT_SECRET: str = os.environ.get(
    "SUPABASE_JWT_SECRET", "paste_from_supabase_project_settings"
)
DEV_INCLUDE_DRAFTS: bool = (
    os.environ.get("DEV_INCLUDE_DRAFTS", "false").lower() == "true"
)

# Sentiment Suggestion Library Settings
SUGGESTION_RECENCY_DAYS: int = int(os.environ.get("SUGGESTION_RECENCY_DAYS", "14"))
SUGGESTION_RECENCY_ENABLED: bool = (
    os.environ.get("SUGGESTION_RECENCY_ENABLED", "true").lower() == "true"
)

# Caregiver Daily Tips Settings
CAREGIVER_DAILY_COUNT: int = int(os.environ.get("CAREGIVER_DAILY_COUNT", "5"))
CAREGIVER_RECENCY_DAYS: int = int(os.environ.get("CAREGIVER_RECENCY_DAYS", "7"))

# Educational Video Library Settings
VIDEO_DIR: str = os.environ.get(
    "VIDEO_DIR", os.path.join(_backend_dir, "data", "videos")
).strip()
VIDEO_MAX_MB_WARN: int = int(os.environ.get("VIDEO_MAX_MB_WARN", "50"))

# Questionnaire Guidance AI Settings
GUIDANCE_AI_ENABLED: bool = (
    os.environ.get("GUIDANCE_AI_ENABLED", "false").lower() == "true"
)
GUIDANCE_AI_REWORD: bool = (
    os.environ.get("GUIDANCE_AI_REWORD", "false").lower() == "true"
)
GUIDANCE_CACHE_DAYS: int = int(os.environ.get("GUIDANCE_CACHE_DAYS", "7"))
GUIDANCE_RATE_LIMIT: str = "20/minute"
GUIDANCE_REQUEST_BODY_BYTES: int = 10 * 1024  # 10 KB body limit for guidance
GUIDANCE_FIXED_THEMES: Set[str] = {
    "sleep",
    "tiredness",
    "baby_health_worry",
    "family_support",
    "loneliness",
    "money",
    "work",
    "self_criticism_guilt",
    "body_physical_symptoms",
    "positive_moments",
    "delivery_fear",
    "pregnancy_symptoms",
    "feeding_difficulty",
    "baby_crying_sleep",
    "recovery_pain",
    "bonding_worries",
    "identity_change",
    "return_to_work",
}

# Gemini Model Name configuration (default: gemini-3.8-flash)
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
SARVAM_API_KEY: str = os.environ.get("SARVAM_API_KEY", "")

# LLM Timeout in seconds
LLM_TIMEOUT: float = 25.0
GUIDANCE_LLM_TIMEOUT: float = 10.0  # 10-second timeout for guidance LLM

# Server configuration
PORT: int = int(os.environ.get("PORT", "3001"))
ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "")
FRONTEND_ORIGIN: str = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

# SQLite Database path
DATABASE_PATH: str = os.environ.get(
    "DATABASE_PATH",
    os.path.join(_backend_dir, "translations.db")
)

# Supported languages for HealHer app
SUPPORTED_LANGUAGES: Set[str] = {"en", "hi", "bn", "ta"}

# Batch limits and security constraints
MAX_BATCH_SIZE: int = 100
MAX_REQUEST_BODY_BYTES: int = 200 * 1024  # 200 KB body limit for translation
SENTIMENT_REQUEST_BODY_BYTES: int = 50 * 1024  # 50 KB body limit for sentiment
MAX_AUDIO_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB limit for audio file uploads
RATE_LIMIT: str = "60/minute"
SENTIMENT_RATE_LIMIT: str = "30/minute"
STT_RATE_LIMIT: str = "30/minute"

# STT Model Configuration
ENABLE_STT: bool = os.environ.get("ENABLE_STT", "false").strip().lower() in ("true", "1", "yes")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
INDIC_CONFORMER_MODEL_ID: str = os.environ.get(
    "INDIC_CONFORMER_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual"
)
INDIC_CONFORMER_DECODING: str = os.environ.get("INDIC_CONFORMER_DECODING", "ctc").strip().lower()
WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "small").strip()
STT_DEVICE: str = os.environ.get("STT_DEVICE", "auto").strip().lower()

# Per-language LLM provider override map
LLM_PROVIDER_BY_LANGUAGE = {
    "en": "gemini",
    "hi": "gemini",
    "bn": "gemini",
    "ta": "gemini",
}


def validate_startup_env() -> None:
    """Validate required environment variables at application startup.

    If GEMINI_API_KEY is missing or still equals placeholder, print
    'GEMINI_API_KEY is missing in .env' and exit.

    If SUPABASE_JWT_SECRET is a placeholder and APP_ENV is not 'development',
    print a clear error and exit.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key or gemini_key in ("paste_your_key_here", "your_key_here"):
        print("GEMINI_API_KEY is missing in .env")
        sys.exit(1)

    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    placeholder_secrets = {
        "paste_from_supabase_project_settings",
        "your_supabase_jwt_secret_here",
        "",
    }

    if app_env != "development" and jwt_secret in placeholder_secrets:
        print(
            "Error: SUPABASE_JWT_SECRET is a placeholder and APP_ENV is not 'development'. "
            "Please configure your real Supabase JWT Secret in .env."
        )
        sys.exit(1)
