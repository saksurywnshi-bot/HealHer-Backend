"""Sentiment Analysis & Pattern Routes for HealHer.

Privacy & Safety:
- Never logs note text or model outputs.
- Never stores raw note text in the database.
- Body size strictly limited to 50 KB.
- Rate limited to 30 requests/minute per user.
- Protected by Supabase JWT with development x-user-ref fallback.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from slowapi.util import get_remote_address

from app.config import SENTIMENT_RATE_LIMIT, SENTIMENT_REQUEST_BODY_BYTES
from app.graphs.sentiment_graph import sentiment_graph
from app.limiter import limiter
from app.services.auth import get_current_user_ref
from app.services.patterns_service import compute_sentiment_patterns
from app.services.sentiment_db import clear_sentiment_results, save_sentiment_result

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])



def get_user_rate_limit_key(request: Request) -> str:
    """Rate limit key based on user authorization header, x-user-ref, or client IP."""
    auth = request.headers.get("Authorization")
    if auth:
        return auth.strip()
    x_user = request.headers.get("x-user-ref")
    if x_user:
        return x_user.strip()
    return get_remote_address(request)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeSentimentRequest(BaseModel):
    note_text: str = Field(
        ...,
        description="User reflection or daily check-in note text.",
        max_length=15000,
    )
    language: Literal["en", "hi", "bn", "ta"] = Field(
        ..., description="Language code of note ('en', 'hi', 'bn', 'ta')."
    )
    stage: Literal["prenatal", "postnatal"] = Field(
        ..., description="Maternal stage: 'prenatal' or 'postnatal'."
    )
    checkin_id: Optional[str] = Field(
        None, description="Optional ID of corresponding checkin record."
    )
    mood_band: Optional[str] = Field(
        None,
        description="Optional clinical mood band (e.g. 'Normal', 'Mild Depressed').",
    )
    anxiety: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Optional clinical anxiety score (0 to 3).",
    )
    sleep_hours: Optional[float] = Field(
        None,
        ge=0.0,
        le=24.0,
        description="Optional logged sleep hours for that day.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "note_text": "I feel so exhausted today and worried about whether the baby is kicking enough.",
                "language": "en",
                "stage": "prenatal",
                "checkin_id": "chk_12345",
                "mood_band": "Normal",
                "anxiety": 1,
                "sleep_hours": 5.5,
            }
        }
    }


class SuggestionItem(BaseModel):
    id: str
    kind: str
    text: str
    text_language: str
    why_this_theme: Optional[str] = None
    why_this_key: Optional[str] = None


class AnalyzeSentimentResponse(BaseModel):
    status: Literal["ok", "too_short", "crisis", "unavailable"]
    tone: Optional[str] = None
    tone_phrase_key: Optional[str] = None
    intensity: Optional[str] = None
    themes: List[str] = []
    theme_keys: List[str] = []
    reflection: Optional[str] = None
    suggestions: List[SuggestionItem] = []
    suggestion: Optional[str] = None  # Kept for backward compatibility
    library_version: Optional[str] = None
    support: bool = False
    mismatch: bool = False
    disclaimer_key: str = "sentiment.disclaimer"
    helplines: Optional[Dict[str, str]] = None
    safe_fallback_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalyzeSentimentResponse,
    summary="Analyze mother's check-in note for tone, themes, and reflection",
)
@limiter.limit(SENTIMENT_RATE_LIMIT, key_func=get_user_rate_limit_key)
async def analyze_sentiment(
    request: Request,
    payload: AnalyzeSentimentRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Analyze note for tone, intensity, themes, and gentle reflection.

    Privacy & Safety:
    - Never logs note text.
    - 50 KB body limit enforced.
    - Rate limited to 30 requests/minute per user.
    """
    # 50 KB body size check
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > SENTIMENT_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body too large. Maximum allowed size for sentiment analysis is {SENTIMENT_REQUEST_BODY_BYTES} bytes.",
        )

    # Check for custom mock function (useful for offline unit testing)
    custom_llm_fn = getattr(request.app.state, "custom_llm_fn", None)

    state_input = {
        "note_text": payload.note_text,
        "language": payload.language,
        "stage": payload.stage,
        "checkin_id": payload.checkin_id,
        "mood_band": payload.mood_band,
        "anxiety": payload.anxiety,
        "sleep_hours": payload.sleep_hours,
        "user_ref": user_ref,
        "custom_llm_fn": custom_llm_fn,
    }

    result = await sentiment_graph.ainvoke(state_input)

    raw_suggestions = result.get("suggestions", [])
    suggestion_items = []
    for s in raw_suggestions:
        if isinstance(s, dict):
            suggestion_items.append(
                SuggestionItem(
                    id=s.get("id", ""),
                    kind=s.get("kind", "do"),
                    text=s.get("text", ""),
                    text_language=s.get("text_language", "en"),
                    why_this_theme=s.get("why_this_theme"),
                    why_this_key=s.get("why_this_key"),
                )
            )

    old_suggestion_str = suggestion_items[0].text if suggestion_items else None

    return AnalyzeSentimentResponse(
        status=result.get("status", "unavailable"),
        tone=result.get("tone"),
        tone_phrase_key=result.get("tone_phrase_key"),
        intensity=result.get("intensity"),
        themes=result.get("themes", []),
        theme_keys=result.get("theme_keys", []),
        reflection=result.get("reflection"),
        suggestions=suggestion_items,
        suggestion=old_suggestion_str,
        library_version=result.get("library_version"),
        support=result.get("support", False),
        mismatch=result.get("mismatch", False),
        disclaimer_key="sentiment.disclaimer",
        helplines=result.get("helplines"),
        safe_fallback_text=result.get("safe_fallback_text"),
    )


@router.delete(
    "/me/suggestion-history",
    summary="Delete suggestion history for authenticated user",
)
def delete_my_suggestion_history(
    user_ref: str = Depends(get_current_user_ref),
):
    """Delete all recorded suggestion history for the authenticated user."""
    from app.services.sentiment_db import delete_user_suggestion_history
    deleted_count = delete_user_suggestion_history(user_ref)
    return {"success": True, "deleted_count": deleted_count}


@router.delete(
    "/results",
    summary="Clear sentiment analysis and reflection records for authenticated user",
)
def clear_my_sentiment_results(
    user_ref: str = Depends(get_current_user_ref),
):
    """Delete all sentiment records and reflection history for this user (useful for resets)."""
    deleted_count = clear_sentiment_results(user_ref)
    return {"success": True, "deleted_count": deleted_count}


@router.get(
    "/patterns",
    summary="Compute longitudinal mood and theme patterns (plain code, no AI)",
)
@limiter.limit(SENTIMENT_RATE_LIMIT, key_func=get_user_rate_limit_key)
def get_sentiment_patterns(
    request: Request,
    period: Literal["week", "month"] = Query(
        "week", description="Analysis period: 'week' (7 days) or 'month' (30 days)."
    ),
    user_ref: str = Depends(get_current_user_ref),
):
    """Compute longitudinal sentiment patterns for the authenticated user.

    Returns:
    - Top 3 themes with counts
    - Difficult tones count
    - Longest consecutive day streak
    - Sleep correlation
    - Change versus previous period
    - suggest_talk_to_someone flag
    - I18n pattern template keys with placeholders
    """
    return compute_sentiment_patterns(user_ref=user_ref, period=period)


class SeedDemoSentimentRequest(BaseModel):
    scenario: Literal["balanced", "tougher", "high_energy", "recovery"] = Field(
        "balanced", description="Scenario type to generate"
    )
    days_count: int = Field(
        30, ge=7, le=90, description="Number of days to distribute notes over"
    )
    stage: Literal["prenatal", "postnatal"] = Field(
        "postnatal", description="Maternal stage"
    )


@router.post(
    "/seed-demo",
    summary="Seed demo sentiment notes distributed across 7-day and 30-day timelines",
)
def seed_demo_sentiment(
    payload: SeedDemoSentimentRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Seed realistic demo sentiment records distributed across a 7-day and 30-day timeline.

    This ensures that when a user switches between 'Past 7 Days' (weekly) and 'Past 30 Days' (monthly),
    they see distinct, realistic distributions, themes, streaks, and sleep correlations.
    """
    # 1. Clear previous sentiment rows for this user
    clear_sentiment_results(user_ref)

    now = datetime.now(timezone.utc)
    scenario = payload.scenario
    days_count = payload.days_count
    stage = payload.stage

    # Define notes template for each scenario
    raw_notes = []

    if scenario == "balanced":
        # 7-day window: 5 reflections
        week_notes = [
            (1, "positive", "low", ["positive_moments", "baby_health_worry"], 7.5, "balanced"),
            (2, "positive", "low", ["positive_moments"], 8.0, "balanced"),
            (3, "neutral", "low", ["sleep", "tiredness"], 7.0, "balanced"),
            (5, "positive", "low", ["positive_moments", "family_support"], 8.0, "balanced"),
            (6, "positive", "low", ["positive_moments"], 7.5, "balanced"),
        ]
        # Days 8..30: 16 reflections
        month_notes = [
            (8, "positive", "low", ["positive_moments", "sleep"], 7.5, "balanced"),
            (10, "neutral", "low", ["tiredness", "positive_moments"], 7.0, "balanced"),
            (11, "positive", "low", ["positive_moments", "family_support"], 8.0, "balanced"),
            (13, "positive", "low", ["positive_moments"], 7.5, "balanced"),
            (14, "neutral", "low", ["sleep"], 7.0, "balanced"),
            (16, "positive", "low", ["positive_moments"], 8.0, "balanced"),
            (17, "positive", "low", ["family_support"], 7.5, "balanced"),
            (19, "tired_overwhelmed", "low", ["tiredness"], 6.0, "balanced"),
            (21, "positive", "low", ["positive_moments"], 8.0, "balanced"),
            (22, "neutral", "low", ["positive_moments"], 7.0, "balanced"),
            (24, "positive", "low", ["family_support", "positive_moments"], 7.5, "balanced"),
            (25, "positive", "low", ["sleep"], 8.0, "balanced"),
            (27, "neutral", "low", ["tiredness"], 7.0, "balanced"),
            (28, "positive", "low", ["positive_moments"], 7.5, "balanced"),
            (29, "positive", "low", ["positive_moments"], 8.0, "balanced"),
            (30, "positive", "low", ["positive_moments", "family_support"], 7.5, "balanced"),
        ]
        raw_notes = week_notes + [n for n in month_notes if n[0] <= days_count]

    elif scenario == "tougher":
        # 7-day window: 6 reflections (5 low/anxious with sleep ~4.5h, 1 balanced)
        week_notes = [
            (1, "low", "medium", ["tiredness", "baby_health_worry"], 4.5, "depressed"),
            (2, "anxious", "high", ["baby_health_worry", "loneliness"], 4.0, "depressed"),
            (3, "low", "medium", ["tiredness", "self_criticism_guilt"], 5.0, "depressed"),
            (4, "anxious", "medium", ["tiredness", "baby_health_worry"], 4.5, "depressed"),
            (5, "tired_overwhelmed", "medium", ["tiredness", "family_support"], 5.0, "depressed"),
            (6, "neutral", "low", ["family_support"], 6.5, "balanced"),
        ]
        # Days 8..30: 17 reflections
        month_notes = [
            (8, "low", "medium", ["tiredness", "baby_health_worry"], 4.5, "depressed"),
            (9, "low", "medium", ["self_criticism_guilt", "tiredness"], 5.0, "depressed"),
            (11, "anxious", "medium", ["baby_health_worry"], 4.5, "depressed"),
            (12, "low", "medium", ["tiredness", "loneliness"], 4.5, "depressed"),
            (14, "tired_overwhelmed", "medium", ["tiredness"], 5.0, "depressed"),
            (15, "neutral", "low", ["family_support"], 7.0, "balanced"),
            (16, "neutral", "low", ["sleep"], 6.5, "balanced"),
            (18, "low", "medium", ["tiredness", "self_criticism_guilt"], 4.5, "depressed"),
            (19, "anxious", "high", ["baby_health_worry"], 4.0, "depressed"),
            (21, "low", "medium", ["tiredness"], 5.0, "depressed"),
            (22, "low", "medium", ["loneliness", "tiredness"], 4.5, "depressed"),
            (24, "tired_overwhelmed", "medium", ["tiredness"], 5.0, "depressed"),
            (25, "anxious", "medium", ["baby_health_worry"], 4.5, "depressed"),
            (26, "low", "medium", ["self_criticism_guilt"], 4.5, "depressed"),
            (28, "neutral", "low", ["family_support"], 6.5, "balanced"),
            (29, "low", "medium", ["tiredness"], 4.5, "depressed"),
            (30, "low", "medium", ["baby_health_worry", "tiredness"], 4.5, "depressed"),
        ]
        raw_notes = week_notes + [n for n in month_notes if n[0] <= days_count]

    elif scenario == "high_energy":
        week_notes = [
            (1, "elevated", "high", ["body_physical_symptoms", "tiredness"], 3.0, "severe_elevated"),
            (2, "elevated", "high", ["tiredness", "sleep"], 2.5, "severe_elevated"),
            (3, "elevated", "medium", ["body_physical_symptoms"], 3.5, "elevated"),
            (4, "elevated", "medium", ["tiredness"], 3.5, "elevated"),
            (5, "neutral", "low", ["positive_moments"], 5.0, "balanced"),
            (6, "elevated", "high", ["body_physical_symptoms"], 3.0, "severe_elevated"),
        ]
        month_notes = [
            (8, "elevated", "medium", ["body_physical_symptoms"], 3.5, "elevated"),
            (10, "neutral", "low", ["family_support"], 6.0, "balanced"),
            (12, "elevated", "medium", ["tiredness"], 3.5, "elevated"),
            (14, "elevated", "high", ["body_physical_symptoms"], 3.0, "severe_elevated"),
            (16, "neutral", "low", ["positive_moments"], 6.0, "balanced"),
            (18, "elevated", "medium", ["body_physical_symptoms"], 4.0, "elevated"),
            (20, "neutral", "low", ["sleep"], 6.0, "balanced"),
            (22, "elevated", "medium", ["tiredness"], 3.5, "elevated"),
            (24, "neutral", "low", ["family_support"], 6.5, "balanced"),
            (26, "elevated", "high", ["body_physical_symptoms"], 3.0, "severe_elevated"),
            (28, "neutral", "low", ["positive_moments"], 6.0, "balanced"),
            (30, "elevated", "medium", ["body_physical_symptoms"], 3.5, "elevated"),
        ]
        raw_notes = week_notes + [n for n in month_notes if n[0] <= days_count]

    else:  # recovery
        week_notes = [
            (1, "positive", "low", ["positive_moments", "family_support"], 7.5, "balanced"),
            (2, "positive", "low", ["positive_moments"], 8.0, "balanced"),
            (3, "neutral", "low", ["sleep"], 7.0, "balanced"),
            (5, "positive", "low", ["positive_moments"], 7.5, "balanced"),
            (6, "neutral", "low", ["family_support"], 7.0, "balanced"),
        ]
        month_notes = [
            (8, "neutral", "low", ["positive_moments"], 6.5, "balanced"),
            (10, "neutral", "low", ["sleep"], 6.5, "balanced"),
            (12, "neutral", "low", ["family_support"], 6.5, "balanced"),
            (14, "low", "medium", ["tiredness"], 5.0, "depressed"),
            (16, "low", "medium", ["tiredness", "baby_health_worry"], 5.0, "depressed"),
            (18, "anxious", "medium", ["baby_health_worry"], 4.5, "depressed"),
            (20, "low", "medium", ["tiredness", "self_criticism_guilt"], 4.5, "depressed"),
            (22, "anxious", "high", ["loneliness"], 4.5, "depressed"),
            (24, "low", "medium", ["tiredness"], 4.5, "depressed"),
            (26, "low", "medium", ["self_criticism_guilt"], 4.5, "depressed"),
            (28, "low", "medium", ["tiredness"], 4.5, "depressed"),
            (30, "anxious", "medium", ["baby_health_worry"], 4.5, "depressed"),
        ]
        raw_notes = week_notes + [n for n in month_notes if n[0] <= days_count]

    # Save to database with exact timestamps
    inserted_count = 0
    for day_offset, tone, intensity, themes, sleep_hours, mood_band in raw_notes:
        note_time = now - timedelta(days=day_offset, hours=2)
        save_sentiment_result(
            user_ref=user_ref,
            stage=stage,
            tone=tone,
            intensity=intensity,
            themes=themes,
            risk_language=False,
            support=(tone in ("low", "anxious", "distressed")),
            mismatch=False,
            sleep_hours=sleep_hours,
            mood_band=mood_band,
            created_at=note_time.isoformat(),
        )
        inserted_count += 1

    return {
        "status": "ok",
        "scenario": scenario,
        "days_count": days_count,
        "inserted_notes": inserted_count,
        "week_notes_count": len([n for n in raw_notes if n[0] <= 7]),
        "month_notes_count": inserted_count,
    }

