"""Admin API Routes for HealHer Translation Management.

Authentication:
Header 'x-admin-token' must match ADMIN_TOKEN.

Endpoints:
- GET   /api/admin/translations?status=machine&language=hi
- PATCH /api/admin/translations
- GET   /api/admin/export.csv
"""

import csv
import io
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, Header, HTTPException, Query, Response, status
from pydantic import BaseModel

from app.config import ADMIN_TOKEN
from app.services.cache import cache_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


def verify_admin_token(x_admin_token: Optional[str] = Header(None, alias="x-admin-token")) -> None:
    """Verify that x-admin-token header matches ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_TOKEN is not configured on the server.",
        )
    if not x_admin_token or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
        )


class UpdateTranslationPayload(BaseModel):
    key: str
    language: Literal["hi", "bn", "ta"]
    translated_text: str


@router.get("/translations", summary="List translations with optional status and language filters")
def list_translations(
    status_filter: Optional[str] = Query(None, alias="status", pattern="^(machine|reviewed)$"),
    language_filter: Optional[str] = Query(None, alias="language", pattern="^(hi|bn|ta)$"),
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """List translations from SQLite cache. Requires valid x-admin-token header."""
    verify_admin_token(x_admin_token)
    records = cache_service.list_translations(status=status_filter, language=language_filter)
    return {"count": len(records), "translations": records}


@router.patch("/translations", summary="Update translation text and mark status as 'reviewed'")
def patch_translation(
    payload: UpdateTranslationPayload,
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Mark a translation as reviewed and update its text.

    Requires valid x-admin-token header.
    """
    verify_admin_token(x_admin_token)

    updated = cache_service.update_reviewed_translation(
        key=payload.key,
        language=payload.language,
        translated_text=payload.translated_text,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Translation for key '{payload.key}' and language '{payload.language}' not found in cache.",
        )

    return {
        "success": True,
        "message": "Translation updated to status 'reviewed'.",
        "translation": updated,
    }


@router.get("/export.csv", summary="Export all translations as CSV")
def export_translations_csv(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Export all cached translations to a CSV file.

    Requires valid x-admin-token header.
    """
    verify_admin_token(x_admin_token)

    rows = cache_service.export_all()
    output = io.StringIO()
    fieldnames = [
        "key",
        "language",
        "source_text",
        "source_hash",
        "translated_text",
        "status",
        "updated_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="translations_export.csv"',
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


class UpdateSuggestionStatusPayload(BaseModel):
    status: Literal["draft", "approved"]


class PatchSentimentSuggestionPayload(BaseModel):
    status: Optional[Literal["draft", "approved"]] = None
    text_en: Optional[str] = None
    text_hi: Optional[str] = None
    text_bn: Optional[str] = None
    text_ta: Optional[str] = None


@router.get("/sentiment-library", summary="All sentiment suggestion library items, sources, and coverage")
def get_sentiment_library(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Return all items with status, evidence_level, source_refs, needs_source, flag_list, and the sources;
    plus coverage per theme and stage counting served items, and themes with fewer than 3 served items.
    """
    verify_admin_token(x_admin_token)
    from app.services.sentiment_library import (
        FIXED_THEMES,
        get_all_library_items_with_overrides,
        load_served_suggestions,
        get_library_version_hash,
    )

    items, sources = get_all_library_items_with_overrides()
    served_items = load_served_suggestions()

    coverage: Dict[str, int] = {}
    for theme in sorted(FIXED_THEMES):
        for stage in ["prenatal", "postnatal"]:
            key = f"{theme}|{stage}"
            count = sum(
                1 for it in served_items
                if (it.get("stage") == stage or it.get("stage") == "both")
                and (theme in it.get("themes", []))
            )
            coverage[key] = count

    thin_themes_prenatal = [
        theme for theme in sorted(FIXED_THEMES)
        if coverage.get(f"{theme}|prenatal", 0) < 3
    ]
    thin_themes_postnatal = [
        theme for theme in sorted(FIXED_THEMES)
        if coverage.get(f"{theme}|postnatal", 0) < 3
    ]

    formatted_items = []
    for it in items:
        formatted_items.append({
            "id": it.get("id"),
            "status": it.get("status"),
            "evidence_level": it.get("evidence_level"),
            "source_refs": it.get("source_refs", []),
            "needs_source": bool(it.get("needs_source")),
            "flag_list": it.get("flag_list", []),
            "stage": it.get("stage"),
            "kind": it.get("kind"),
            "tones": it.get("tones", []),
            "themes": it.get("themes", []),
            "effort": it.get("effort"),
            "generic": bool(it.get("generic")),
            "text_en": it.get("text_en"),
            "text_hi": it.get("text_hi", ""),
            "text_bn": it.get("text_bn", ""),
            "text_ta": it.get("text_ta", ""),
        })

    return {
        "items": formatted_items,
        "sources": sources,
        "coverage": coverage,
        "thin_coverage": {
            "prenatal": thin_themes_prenatal,
            "postnatal": thin_themes_postnatal,
        },
        "total_items": len(items),
        "total_served": len(served_items),
        "library_version": get_library_version_hash(),
    }


@router.patch("/sentiment-library/suggestion/{suggestion_id}", summary="Set status or edit wording into sentiment_overrides.json")
def patch_sentiment_library_suggestion(
    suggestion_id: str,
    payload: PatchSentimentSuggestionPayload,
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Set status 'draft' or 'approved' (refuses 'approved' if needs_source or source_refs empty);
    allows editing text_hi, text_bn, text_ta (and text_en wording) into data/sentiment_overrides.json; re-validates.
    """
    verify_admin_token(x_admin_token)
    from app.services.sentiment_library import save_override, get_library_version_hash

    edits = payload.model_dump(exclude_unset=True)
    if not edits:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    try:
        updated = save_override(suggestion_id, edits)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return {
        "success": True,
        "message": f"Suggestion '{suggestion_id}' successfully updated.",
        "suggestion": updated,
        "library_version": get_library_version_hash(),
    }


@router.get("/sentiment-library/untranslated", summary="Items missing hi, bn or ta text")
def get_sentiment_library_untranslated(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Return items missing hi, bn or ta text."""
    verify_admin_token(x_admin_token)
    from app.services.sentiment_library import get_all_library_items_with_overrides

    items, _ = get_all_library_items_with_overrides()
    untranslated = []
    for it in items:
        missing_langs = []
        for lang in ["hi", "bn", "ta"]:
            txt = (it.get(f"text_{lang}") or "").strip()
            if not txt or txt.startswith("TODO"):
                missing_langs.append(lang)
        if missing_langs:
            untranslated.append({
                "id": it.get("id"),
                "missing_languages": missing_langs,
                "text_en": it.get("text_en"),
                "stage": it.get("stage"),
                "status": it.get("status"),
            })

    return {
        "count": len(untranslated),
        "untranslated": untranslated,
    }


@router.get("/sentiment-library/stats", summary="Telemetry counts for sentiment suggestion library")
def get_sentiment_library_telemetry_stats(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Counts only: how often each suggestion id was shown in the last 30 days,
    repeat rate, and number of results with no suggestions. No user identifiers.
    """
    verify_admin_token(x_admin_token)
    from app.services.sentiment_db import get_sentiment_library_stats

    return get_sentiment_library_stats(days_shown=30, repeat_window_days=14)


@router.get("/suggestions", summary="List all suggestions in the library with their statuses")
def list_suggestions(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """List all suggestions from the library. Requires valid x-admin-token header."""
    verify_admin_token(x_admin_token)
    from app.services.suggestions_service import get_all_suggestions

    items = get_all_suggestions()
    return {"count": len(items), "suggestions": items}


@router.patch("/suggestions/{suggestion_id}", summary="Update the status of a suggestion (e.g. approve after review)")
def patch_suggestion_status(
    suggestion_id: str,
    payload: UpdateSuggestionStatusPayload,
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Update status ('draft' or 'approved') of a specific suggestion after clinical review."""
    verify_admin_token(x_admin_token)
    from app.services.suggestions_service import update_suggestion_status

    try:
        updated = update_suggestion_status(suggestion_id, payload.status)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suggestion with id '{suggestion_id}' not found.",
        )

    return {
        "success": True,
        "message": f"Suggestion '{suggestion_id}' status updated to '{payload.status}'.",
        "suggestion": updated,
    }


@router.get("/guidance/stats", summary="Guidance AI Telemetry and Validation Stats (Counts Only)")
def get_guidance_telemetry_stats(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Returns counts of AI selections used, fallbacks, validation failures by type, and cache hits.

    Strict Privacy Rule: Contains counts only, absolutely no user text or model content.
    """
    verify_admin_token(x_admin_token)
    from app.services.guidance_db import get_guidance_stats

    stats = get_guidance_stats()
    return stats


@router.get("/guidance/flags", summary="Get current values of Guidance AI feature flags")
def get_guidance_feature_flags(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Read-only view of GUIDANCE_AI_ENABLED and GUIDANCE_AI_REWORD feature flags."""
    verify_admin_token(x_admin_token)
    import os
    from app.config import (
        GUIDANCE_AI_ENABLED,
        GUIDANCE_AI_REWORD,
        GUIDANCE_CACHE_DAYS,
    )

    ai_enabled = os.environ.get("GUIDANCE_AI_ENABLED", str(GUIDANCE_AI_ENABLED)).lower() == "true"
    ai_reword = os.environ.get("GUIDANCE_AI_REWORD", str(GUIDANCE_AI_REWORD)).lower() == "true"
    cache_days = int(os.environ.get("GUIDANCE_CACHE_DAYS", str(GUIDANCE_CACHE_DAYS)))

    return {
        "guidance_ai_enabled": ai_enabled,
        "guidance_ai_reword": ai_reword,
        "guidance_cache_days": cache_days,
    }
