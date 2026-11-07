"""FastAPI Main Application for HealHer Backend.

Services:
- Translation API (/api/translate)
- Sentiment Analysis API (/api/sentiment/analyze, /api/sentiment/patterns)
- Admin Suggestions & Translations Management (/api/admin/*)

Privacy & Security:
- Tracing disabled: LANGCHAIN_TRACING_V2=false, LANGSMITH_TRACING=false.
- CORS restricted to FRONTEND_ORIGIN.
- Rate limiting applied per IP / user.
- Request body limits strictly enforced (200 KB general, 50 KB sentiment).
- No logging of request/response bodies, notes, or secrets.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.services.auth import get_current_user_ref

from app.config import (
    validate_startup_env,
    FRONTEND_ORIGIN,
    MAX_REQUEST_BODY_BYTES,
    MAX_AUDIO_SIZE_BYTES,
    GUIDANCE_REQUEST_BODY_BYTES,
    APP_ENV,
    ENABLE_STT,
)
from app.limiter import limiter
from app.routes import translate, admin, sentiment, transcribe, insights, questionnaire, tts, caregiver, videos
from app.graphs.translate_graph import get_translation_graph_mermaid
from app.graphs.sentiment_graph import get_sentiment_graph_mermaid
from app.services.sentiment_db import init_sentiment_db
from app.services.sentiment_library import (
    load_raw_library,
    get_all_library_items_with_overrides,
    load_served_suggestions,
)
from app.services.insights.storage import init_insights_db
from app.services.guidance_db import init_guidance_db
from app.services.caregiver_db import init_caregiver_db, delete_user_caregiver_history
from app.services.caregiver_library import (
    load_caregiver_library,
    get_raw_caregiver_library,
    get_served_caregiver_library,
)
from app.services.video_library import initialize_video_library

# Configure safe logging (no bodies, no keys)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healher_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: validates environment, validates suggestion library, and initializes databases on startup."""
    validate_startup_env()
    load_raw_library()
    load_caregiver_library()
    initialize_video_library()
    init_sentiment_db()
    init_insights_db()
    init_guidance_db()
    init_caregiver_db()

    # Log operational status without secrets
    sentiment_items, _ = get_all_library_items_with_overrides()
    sentiment_approved = sum(1 for item in sentiment_items if item.get("status") == "approved")
    sentiment_served = len(load_served_suggestions())

    cg_raw = get_raw_caregiver_library()
    cg_suggestions = cg_raw.get("suggestions", [])
    cg_approved = sum(1 for item in cg_suggestions if item.get("status") == "approved")
    cg_served = len((get_served_caregiver_library() or {}).get("suggestions", []))

    logger.info(
        "Startup Status: APP_ENV=%s, ENABLE_STT=%s | "
        "Sentiment Library: %d approved (%d served / %d total) | "
        "Caregiver Library: %d approved (%d served / %d total)",
        APP_ENV,
        ENABLE_STT,
        sentiment_approved,
        sentiment_served,
        len(sentiment_items),
        cg_approved,
        cg_served,
        len(cg_suggestions),
    )
    logger.info("HealHer backend service initialized successfully.")
    yield
    logger.info("HealHer backend service shutting down.")


# 200 KB General Body Limit Middleware, with 25 MB for STT audio and 10 KB for questionnaire guidance
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_bytes: int = MAX_REQUEST_BODY_BYTES,
        max_audio_bytes: int = MAX_AUDIO_SIZE_BYTES,
        max_guidance_bytes: int = GUIDANCE_REQUEST_BODY_BYTES,
    ):
        super().__init__(app)
        self.max_bytes = max_bytes
        self.max_audio_bytes = max_audio_bytes
        self.max_guidance_bytes = max_guidance_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if request.url.path.startswith("/api/transcribe"):
                    limit = self.max_audio_bytes
                elif request.url.path.startswith("/api/questionnaire"):
                    limit = self.max_guidance_bytes
                else:
                    limit = self.max_bytes

                if int(content_length) > limit:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Request body too large. Maximum allowed size is {limit} bytes."
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


app = FastAPI(
    title="HealHer Mental Health Backend",
    description="Translation and sentiment-analysis backend for perinatal and postpartum mental health in India.",
    version="1.1.0",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security Middlewares
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
cors_origins = list({FRONTEND_ORIGIN, "http://localhost:5173", "http://localhost:8080", "http://localhost:8081", "http://127.0.0.1:8080", "http://127.0.0.1:5173"})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(translate.router)
app.include_router(sentiment.router)
app.include_router(transcribe.router)
app.include_router(insights.router)
app.include_router(questionnaire.router)
app.include_router(admin.router)
app.include_router(tts.router)
app.include_router(caregiver.router)
app.include_router(videos.router)


@app.delete("/api/me/suggestion-history", tags=["sentiment"])
async def delete_my_suggestion_history_root(
    user_ref: str = Depends(get_current_user_ref),
):
    """Delete suggestion history for authenticated user (root path alias)."""
    from app.services.sentiment_db import delete_user_suggestion_history

    deleted_count = delete_user_suggestion_history(user_ref)
    caregiver_deleted = delete_user_caregiver_history(user_ref)
    return {
        "success": True,
        "deleted_count": deleted_count,
        "caregiver_deleted_count": caregiver_deleted,
    }


@app.get("/health", tags=["system"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "HealHer Mental Health Service"}


@app.get("/api/graph/mermaid", response_class=PlainTextResponse, tags=["system"])
def translation_graph_mermaid():
    """Return the Translation LangGraph workflow diagram as Mermaid markup."""
    return get_translation_graph_mermaid()


@app.get("/api/graph/sentiment/mermaid", response_class=PlainTextResponse, tags=["system"])
def sentiment_graph_mermaid():
    """Return the Sentiment Analysis LangGraph workflow diagram as Mermaid markup."""
    return get_sentiment_graph_mermaid()
