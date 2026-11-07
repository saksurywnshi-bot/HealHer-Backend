"""Theme-specific, tone-aware, stage-aware Suggestion Selector (Plain Code, No AI).

Inputs:
- stage ('prenatal' | 'postnatal')
- tone (validated from model: positive, neutral, tired_overwhelmed, anxious, low, distressed)
- themes (ordered, from model, validated)
- escalation ('none', 'see_clinician', 'see_clinician_soon', 'see_clinician_urgent', etc.)
- risk_flag (bool, risk_language)
- user_ref (str, SHA-256 user identity)
- language (str: 'en', 'hi', 'bn', 'ta')
- today_date (optional date or YYYY-MM-DD str for deterministic evaluation)

Selection Rules:
1. If tone is 'distressed', risk_language is True, or escalation != 'none': return NO library tips.
2. Pool = served items where (item.stage == stage OR item.stage == 'both')
   AND tone in item.tones
   AND (item.generic is True OR at least one of item.themes in detected themes).
3. Primary theme = first detected theme with at least one qualifying non-generic item.
   If themes is empty, use generic items only.
4. Scoring:
   - Primary theme match: +3
   - Secondary detected theme match: +1
   - Tone is first entry in item.tones: +2
   - item.stage == stage (not 'both'): +1
   - Not shown in last SUGGESTION_RECENCY_DAYS days: +2
   - Effort fit: if tone == 'tired_overwhelmed':
       effort in (none, 1min, 5min, talk): +1
       effort == 'longer': -2
5. Deterministic tie-breaker seeded by SHA-256(user_ref + local date).
6. Build up to 3 items:
   - Best 'understand' item first if one exists.
   - Then best remaining 'do' or 'talk' or 'clinician'.
   - At most ONE kind 'clinician' per result.
   - At most ONE generic item, and ONLY if fewer than 3 theme-specific items qualified.
   - Never repeat an item id.
   - Prefer variety of themes only when several themes were detected, otherwise stay on primary theme.
7. Recency: if SUGGESTION_RECENCY_ENABLED=true, exclude ids shown inside window when enough
   alternatives remain; if pool drops below number needed, allow least recently shown ids.
   Store each shown id in SQLite suggestion_history.
8. Language: text_<language> if non-empty, else text_en with text_language='en'.
9. why_this: "Because you wrote about {theme_label}", i18n key suggestion.why.<theme>.
   Omitted for generic items.
10. If pool is empty, return suggestions: [].
"""

import hashlib
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import (
    SUGGESTION_RECENCY_DAYS,
    SUGGESTION_RECENCY_ENABLED,
)
from app.services.sentiment_db import (
    get_recently_shown_ids,
    get_suggestion_last_shown_times,
    record_shown_suggestions,
)
from app.services.sentiment_library import (
    get_theme_labels,
    load_served_suggestions,
)

logger = logging.getLogger("healher_backend.suggestion_selector")


def compute_item_score(
    item: Dict[str, Any],
    stage: str,
    tone: str,
    primary_theme: Optional[str],
    detected_themes: List[str],
    recently_shown_ids: Set[str],
) -> int:
    """Calculate score for an item according to Rule 4."""
    score = 0
    item_themes = item.get("themes", [])

    # Primary theme match +3
    if primary_theme and primary_theme in item_themes:
        score += 3

    # Secondary detected theme match +1
    secondary_themes = [t for t in detected_themes if t != primary_theme]
    if any(t in item_themes for t in secondary_themes):
        score += 1

    # Tone is first entry in item.tones +2
    tones = item.get("tones", [])
    if tones and tones[0] == tone:
        score += 2

    # item.stage equals stage (not 'both') +1
    if item.get("stage") == stage:
        score += 1

    # Not shown in last SUGGESTION_RECENCY_DAYS days +2
    item_id = item.get("id", "")
    if item_id not in recently_shown_ids:
        score += 2

    # Effort fit for 'tired_overwhelmed' tone
    if tone == "tired_overwhelmed":
        effort = item.get("effort", "")
        if effort in {"none", "1min", "5min", "talk"}:
            score += 1
        elif effort == "longer":
            score -= 2

    return score


def get_deterministic_tie_breaker(user_ref: str, date_str: str, item_id: str) -> str:
    """Deterministic tie-breaker hash based on SHA-256(user_ref + date + item_id)."""
    seed_str = f"{user_ref}:{date_str}:{item_id}"
    return hashlib.sha256(seed_str.encode("utf-8")).hexdigest()


def select_sentiment_suggestions(
    stage: str,
    tone: str,
    themes: List[str],
    escalation: str = "none",
    risk_language: bool = False,
    user_ref: str = "",
    language: str = "en",
    today_date: Optional[Any] = None,
    dev_include_drafts: Optional[bool] = None,
    app_env: Optional[str] = None,
    record_history: bool = True,
) -> List[Dict[str, Any]]:
    """Select up to 3 clinically approved suggestions using plain code rules."""
    # -------------------------------------------------------------------------
    # Rule 1: No library tips for distressed tone, risk_language, or escalation
    # -------------------------------------------------------------------------
    if (
        tone == "distressed"
        or risk_language
        or (escalation and escalation.lower() != "none")
    ):
        return []

    # Format local date string
    if today_date is None:
        date_str = date.today().isoformat()
    elif isinstance(today_date, (date, datetime)):
        date_str = today_date.strftime("%Y-%m-%d")
    else:
        date_str = str(today_date)

    # Load served library items
    served_items = load_served_suggestions(
        dev_include_drafts=dev_include_drafts, app_env=app_env
    )
    if not served_items:
        return []

    # -------------------------------------------------------------------------
    # Rule 2: Candidate Pool
    # -------------------------------------------------------------------------
    detected_themes = [t for t in (themes or []) if isinstance(t, str)]
    detected_themes_set = set(detected_themes)

    initial_pool: List[Dict[str, Any]] = []
    for item in served_items:
        item_stage = item.get("stage", "both")
        if item_stage != stage and item_stage != "both":
            continue

        item_tones = item.get("tones", [])
        if tone not in item_tones:
            continue

        is_generic = bool(item.get("generic"))
        item_themes = item.get("themes", [])
        has_theme_match = bool(detected_themes_set.intersection(set(item_themes)))

        if is_generic or has_theme_match:
            initial_pool.append(item)

    if not initial_pool:
        return []

    # -------------------------------------------------------------------------
    # Rule 3: Primary Theme
    # -------------------------------------------------------------------------
    primary_theme: Optional[str] = None
    for t in detected_themes:
        # Check if any non-generic item in initial_pool has theme t
        has_item = any(
            (not it.get("generic")) and (t in it.get("themes", []))
            for it in initial_pool
        )
        if has_item:
            primary_theme = t
            break

    # Count how many theme-specific items qualified in the pool
    theme_specific_pool = [it for it in initial_pool if not it.get("generic")]
    num_theme_specific_qualified = len(theme_specific_pool)

    # -------------------------------------------------------------------------
    # Rule 7: Recency Filtering
    # -------------------------------------------------------------------------
    recently_shown_ids_set: Set[str] = set()
    last_shown_times: Dict[str, str] = {}

    if SUGGESTION_RECENCY_ENABLED and user_ref:
        recently_shown_ids_set = set(
            get_recently_shown_ids(user_ref, days=SUGGESTION_RECENCY_DAYS)
        )
        last_shown_times = get_suggestion_last_shown_times(
            user_ref, days=SUGGESTION_RECENCY_DAYS
        )

    # Check if excluding recently shown items leaves enough alternatives
    # Needed count is up to 3 items
    needed_count = min(3, len(initial_pool))
    non_recent_pool = [
        it for it in initial_pool if it.get("id") not in recently_shown_ids_set
    ]

    if not SUGGESTION_RECENCY_ENABLED or not user_ref:
        active_pool = initial_pool
    elif len(non_recent_pool) >= needed_count:
        active_pool = non_recent_pool
    else:
        # Pool would drop below needed count: allow least recently shown items back
        recent_in_pool = [
            it for it in initial_pool if it.get("id") in recently_shown_ids_set
        ]
        # Sort recent items by shown_at ascending (oldest first)
        recent_in_pool.sort(
            key=lambda it: last_shown_times.get(it.get("id", ""), "")
        )
        shortage = needed_count - len(non_recent_pool)
        allowed_back = recent_in_pool[:shortage]
        active_pool = non_recent_pool + allowed_back

    if not active_pool:
        return []

    # -------------------------------------------------------------------------
    # Rule 4 & 5: Scoring and Deterministic Sorting
    # -------------------------------------------------------------------------
    scored_items: List[Tuple[int, str, Dict[str, Any]]] = []
    for item in active_pool:
        score = compute_item_score(
            item=item,
            stage=stage,
            tone=tone,
            primary_theme=primary_theme,
            detected_themes=detected_themes,
            recently_shown_ids=recently_shown_ids_set,
        )
        tie_breaker = get_deterministic_tie_breaker(user_ref, date_str, item.get("id", ""))
        scored_items.append((score, tie_breaker, item))

    # Sort descending by score, then by deterministic tie-breaker hash
    scored_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    sorted_candidates = [x[2] for x in scored_items]

    # -------------------------------------------------------------------------
    # Rule 6: Build the result of up to 3 items
    # -------------------------------------------------------------------------
    # Constraints:
    # 1. First the best item with kind 'understand' (normalising) if one exists.
    # 2. Then best remaining items of kind 'do' or 'talk' (or clinician).
    # 3. At most ONE item of kind 'clinician' per result.
    # 4. At most ONE generic item, and ONLY if fewer than 3 theme-specific items qualified.
    # 5. Never repeat an item id.
    # 6. Prefer variety of themes only when several themes were detected, otherwise stay on primary theme.
    # 7. Theme-first: with themes [feeding_difficulty, sleep], first non-understand item belongs to feeding_difficulty.
    allow_generic = (num_theme_specific_qualified < 3)

    selected_items: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    selected_themes: Set[str] = set()
    has_clinician = False
    has_generic = False

    def can_pick(cand: Dict[str, Any]) -> bool:
        cid = cand.get("id", "")
        if cid in selected_ids:
            return False
        if cand.get("kind") == "clinician" and has_clinician:
            return False
        if cand.get("generic"):
            if not allow_generic or has_generic:
                return False
        return True

    # Slot 1: Best 'understand' item if one exists
    understand_candidates = [
        c for c in sorted_candidates if c.get("kind") == "understand" and can_pick(c)
    ]
    if understand_candidates:
        # If primary theme exists, prefer understand candidate matching primary theme
        primary_understand = [
            c for c in understand_candidates if primary_theme in c.get("themes", [])
        ]
        best_u = primary_understand[0] if primary_understand else understand_candidates[0]
        selected_items.append(best_u)
        selected_ids.add(best_u.get("id", ""))
        if best_u.get("kind") == "clinician":
            has_clinician = True
        if best_u.get("generic"):
            has_generic = True
        for th in best_u.get("themes", []):
            if th in detected_themes_set:
                selected_themes.add(th)

    # Slot 2 & 3: Action / Talk / Clinician items (strictly non-understand)
    # Rule: Theme-first: first non-understand item must belong to primary theme if possible!
    # And if several themes detected, prefer variety for the 3rd item.
    while len(selected_items) < 3:
        remaining_candidates = [c for c in sorted_candidates if can_pick(c) and c.get("kind") != "understand"]
        if not remaining_candidates:
            # Fallback if no non-understand candidates remain
            remaining_candidates = [c for c in sorted_candidates if can_pick(c)]
            if not remaining_candidates:
                break

        is_first_non_understand = (
            len(selected_items) == 0
            or (len(selected_items) == 1 and selected_items[0].get("kind") == "understand")
        )

        chosen_candidate: Optional[Dict[str, Any]] = None

        if is_first_non_understand and primary_theme:
            # Theme-first: First non-understand item should match the primary theme!
            primary_matches = [
                c for c in remaining_candidates if primary_theme in c.get("themes", [])
            ]
            if primary_matches:
                chosen_candidate = primary_matches[0]

        if not chosen_candidate:
            # If several themes were detected, prefer an item matching an unrepresented detected theme
            if len(detected_themes) > 1:
                unrepresented_themes = [
                    t for t in detected_themes if t not in selected_themes
                ]
                variety_matches = [
                    c
                    for c in remaining_candidates
                    if any(t in c.get("themes", []) for t in unrepresented_themes)
                ]
                if variety_matches:
                    chosen_candidate = variety_matches[0]

        if not chosen_candidate:
            # If only 1 theme detected, prefer staying on primary theme if available
            if primary_theme:
                primary_matches = [
                    c for c in remaining_candidates if primary_theme in c.get("themes", [])
                ]
                if primary_matches:
                    chosen_candidate = primary_matches[0]

        if not chosen_candidate:
            # Take highest ranked candidate available
            chosen_candidate = remaining_candidates[0]

        selected_items.append(chosen_candidate)
        selected_ids.add(chosen_candidate.get("id", ""))
        if chosen_candidate.get("kind") == "clinician":
            has_clinician = True
        if chosen_candidate.get("generic"):
            has_generic = True
        for th in chosen_candidate.get("themes", []):
            if th in detected_themes_set:
                selected_themes.add(th)

    # -------------------------------------------------------------------------
    # Record history if enabled
    # -------------------------------------------------------------------------
    if record_history and user_ref and selected_items:
        record_shown_suggestions(
            user_ref=user_ref,
            suggestion_ids=[item["id"] for item in selected_items],
        )

    # -------------------------------------------------------------------------
    # Rule 8 & 9: Format Response items with language and why_this
    # -------------------------------------------------------------------------
    theme_labels = get_theme_labels()
    formatted_results: List[Dict[str, Any]] = []

    for item in selected_items:
        item_id = item.get("id", "")
        kind = item.get("kind", "do")

        # Language selection (Rule 8)
        lang_key = f"text_{language}"
        lang_text = (item.get(lang_key) or "").strip()
        if language in ("hi", "bn", "ta") and lang_text and not lang_text.startswith("TODO"):
            text = lang_text
            text_language = language
        else:
            text = (item.get("text_en") or "").strip()
            text_language = "en"

        # why_this selection (Rule 9)
        why_this_theme: Optional[str] = None
        why_this_key: Optional[str] = None

        if not item.get("generic"):
            item_themes = item.get("themes", [])
            # Find matched theme (prefer primary theme, then first detected theme match)
            matched_t: Optional[str] = None
            if primary_theme and primary_theme in item_themes:
                matched_t = primary_theme
            else:
                for t in detected_themes:
                    if t in item_themes:
                        matched_t = t
                        break
                if not matched_t and item_themes:
                    matched_t = item_themes[0]

            if matched_t:
                label_val = theme_labels.get(matched_t, matched_t)
                why_this_theme = f"Because you wrote about {label_val}"
                why_this_key = f"suggestion.why.{matched_t}"

        res_item: Dict[str, Any] = {
            "id": item_id,
            "kind": kind,
            "text": text,
            "text_language": text_language,
        }
        if why_this_theme is not None:
            res_item["why_this_theme"] = why_this_theme
            res_item["why_this_key"] = why_this_key
        else:
            res_item["why_this_theme"] = None
            res_item["why_this_key"] = None

        formatted_results.append(res_item)

    return formatted_results
