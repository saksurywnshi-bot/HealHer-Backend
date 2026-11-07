"""Service for managing and selecting clinically approved suggestions.

DEPRECATION NOTE:
The old data/suggestions.json and config/theme_to_suggestions.json have been moved to legacy/.
This module provides backward-compatible adapters delegating to:
- app.services.sentiment_library
- app.services.suggestion_selector
"""

from typing import Any, Dict, List, Optional
from app.services.sentiment_library import (
    load_served_suggestions,
    get_all_library_items_with_overrides,
    save_override,
)
from app.services.suggestion_selector import select_sentiment_suggestions


def load_suggestions() -> List[Dict[str, Any]]:
    """Load suggestions from new library (backward-compatible)."""
    return load_served_suggestions()


def get_all_suggestions() -> List[Dict[str, Any]]:
    """Return all suggestions in the library (backward-compatible)."""
    items, _ = get_all_library_items_with_overrides()
    return items


def update_suggestion_status(
    suggestion_id: str, new_status: str
) -> Optional[Dict[str, Any]]:
    """Update status of a specific suggestion via sentiment_overrides.json."""
    if new_status not in ("draft", "approved"):
        raise ValueError("Invalid status. Must be 'draft' or 'approved'.")

    return save_override(suggestion_id, {"status": new_status})


def pick_suggestions(
    tone: str,
    themes: List[str],
    stage: str,
    language: str = "en",
    dev_include_drafts: Optional[bool] = None,
    user_ref: str = "",
) -> List[Dict[str, Any]]:
    """Select suggestions delegating to theme-specific, tone-aware, stage-aware selector."""
    return select_sentiment_suggestions(
        stage=stage,
        tone=tone,
        themes=themes,
        language=language,
        user_ref=user_ref,
        dev_include_drafts=dev_include_drafts,
        record_history=False,
    )
