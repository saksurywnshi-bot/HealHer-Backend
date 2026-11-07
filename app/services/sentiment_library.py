"""Sentiment Suggestion Library Loader and Validator.

Rules:
- Loads data/sentiment_suggestions.json once at startup and validates it.
- Refuses to start with a clear error on:
  * invalid source_refs (not in sources)
  * theme not in fixed 18
  * tone not in fixed 6
  * approved item with needs_source=True or empty source_refs
  * empty text_en
  * digits in text_en
  * forbidden words in text_en
  * medication words in text_en
- Serves only status 'approved', UNLESS DEV_INCLUDE_DRAFTS=true AND APP_ENV=development.
- Merges clinician edits from data/sentiment_overrides.json on top and re-validates.
- Never modifies data/sentiment_suggestions.json.
- Exposes a library version hash.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import APP_ENV, DEV_INCLUDE_DRAFTS

logger = logging.getLogger("healher_backend.sentiment_library")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

SUGGESTIONS_PATH = os.path.join(DATA_DIR, "sentiment_suggestions.json")
OVERRIDES_PATH = os.path.join(DATA_DIR, "sentiment_overrides.json")
FORBIDDEN_WORDS_PATH = os.path.join(CONFIG_DIR, "forbidden_words.json")
MEDICATION_WORDS_PATH = os.path.join(CONFIG_DIR, "medication_words.json")

FIXED_THEMES: Set[str] = {
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

FIXED_TONES: Set[str] = {
    "positive",
    "neutral",
    "tired_overwhelmed",
    "anxious",
    "low",
    "distressed",
}

VALID_STAGES: Set[str] = {"prenatal", "postnatal", "both"}


def _load_json_file(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_forbidden_words() -> List[str]:
    data = _load_json_file(FORBIDDEN_WORDS_PATH)
    words: List[str] = []
    for k, v in data.items():
        if not k.startswith("_") and isinstance(v, list):
            words.extend(v)
    return words


def load_medication_words() -> List[str]:
    data = _load_json_file(MEDICATION_WORDS_PATH)
    words: List[str] = []
    for k, v in data.items():
        if not k.startswith("_") and isinstance(v, list):
            words.extend(v)
    return words


def check_text_violations(text: str) -> Optional[str]:
    """Check text for digits, forbidden terms, and medication terms."""
    if not text:
        return "empty_text"

    # Digits check
    if re.search(r"\d", text):
        return f"contains_digit: '{text}'"

    text_lower = text.lower()

    # Forbidden words check
    for word in load_forbidden_words():
        w_clean = word.strip().lower()
        if not w_clean:
            continue
        # Use word prefix/boundary matching
        pattern = rf"\b{re.escape(w_clean)}"
        if re.search(pattern, text_lower):
            return f"contains_forbidden_word: '{w_clean}' in '{text}'"

    # Medication words check
    for med in load_medication_words():
        m_clean = med.strip().lower()
        if not m_clean:
            continue
        pattern = rf"\b{re.escape(m_clean)}\b"
        if re.search(pattern, text_lower):
            return f"contains_medication_word: '{m_clean}' in '{text}'"

    return None


def validate_suggestion_item(item: Dict[str, Any], sources: Dict[str, Any]) -> None:
    """Validate a single suggestion item according to clinical rules."""
    item_id = item.get("id")
    if not item_id:
        raise ValueError("Suggestion item missing 'id'.")

    # 1. Source reference check
    source_refs = item.get("source_refs", [])
    for src in source_refs:
        if src not in sources:
            raise ValueError(
                f"Suggestion '{item_id}' refers to unknown source_ref '{src}' (not in sources)."
            )

    # 2. Themes check
    themes = item.get("themes", [])
    for theme in themes:
        if theme not in FIXED_THEMES:
            raise ValueError(
                f"Suggestion '{item_id}' contains invalid theme '{theme}'. Must be one of {sorted(FIXED_THEMES)}."
            )

    # 3. Tones check
    tones = item.get("tones", [])
    if not tones:
        raise ValueError(f"Suggestion '{item_id}' has empty tones list.")
    for tone in tones:
        if tone not in FIXED_TONES:
            raise ValueError(
                f"Suggestion '{item_id}' contains invalid tone '{tone}'. Must be one of {sorted(FIXED_TONES)}."
            )

    # 4. Status and needs_source check
    status = item.get("status", "draft")
    needs_source = item.get("needs_source", False)
    if status == "approved":
        if needs_source is True or not source_refs:
            raise ValueError(
                f"Suggestion '{item_id}' has status 'approved' but needs_source is True or source_refs is empty."
            )

    # 5. English text checks
    text_en = item.get("text_en", "")
    if not text_en or not text_en.strip():
        raise ValueError(f"Suggestion '{item_id}' has empty text_en.")

    violation = check_text_violations(text_en)
    if violation:
        raise ValueError(
            f"Suggestion '{item_id}' failed text validation in text_en: {violation}."
        )


def validate_library_data(data: Dict[str, Any]) -> None:
    """Validate entire library structure and all suggestions."""
    if not isinstance(data, dict):
        raise ValueError("Library data must be a JSON dictionary.")

    sources = data.get("sources", {})
    suggestions = data.get("suggestions", [])
    if not suggestions:
        raise ValueError("Library data has no 'suggestions'.")

    for item in suggestions:
        validate_suggestion_item(item, sources)


def get_library_version_hash() -> str:
    """Compute deterministic SHA-256 hash of suggestions library and overrides."""
    hasher = hashlib.sha256()

    if os.path.exists(SUGGESTIONS_PATH):
        with open(SUGGESTIONS_PATH, "rb") as f:
            hasher.update(f.read())
    else:
        hasher.update(b"{}")

    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, "rb") as f:
            hasher.update(f.read())
    else:
        hasher.update(b"{}")

    return hasher.hexdigest()[:16]


def load_raw_library() -> Dict[str, Any]:
    """Load raw library from data/sentiment_suggestions.json and validate it once."""
    if not os.path.exists(SUGGESTIONS_PATH):
        raise FileNotFoundError(f"Sentiment suggestions file not found at: {SUGGESTIONS_PATH}")

    with open(SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_library_data(data)
    return data


def load_overrides() -> Dict[str, Dict[str, Any]]:
    """Load clinician overrides from data/sentiment_overrides.json."""
    if not os.path.exists(OVERRIDES_PATH):
        return {}
    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load sentiment_overrides.json: %s", exc)
        return {}


def save_override(suggestion_id: str, edits: Dict[str, Any]) -> Dict[str, Any]:
    """Save clinician edits to data/sentiment_overrides.json and re-validate."""
    raw = load_raw_library()
    sources = raw.get("sources", {})
    all_suggestions = {s["id"]: s for s in raw.get("suggestions", [])}

    if suggestion_id not in all_suggestions:
        raise ValueError(f"Suggestion id '{suggestion_id}' not found in library.")

    base_item = all_suggestions[suggestion_id]
    current_overrides = load_overrides()
    item_overrides = current_overrides.get(suggestion_id, {})

    # Permitted override fields: status, text_en, text_hi, text_bn, text_ta
    allowed_fields = {"status", "text_en", "text_hi", "text_bn", "text_ta"}
    for k, v in edits.items():
        if k in allowed_fields and v is not None:
            item_overrides[k] = v

    # Construct merged candidate to validate
    merged_candidate = {**base_item, **item_overrides}

    # Verify approved status constraints
    if merged_candidate.get("status") == "approved":
        if merged_candidate.get("needs_source") is True or not merged_candidate.get("source_refs"):
            raise ValueError(
                f"Cannot approve suggestion '{suggestion_id}': needs_source is True or source_refs is empty."
            )

    # Validate merged item
    validate_suggestion_item(merged_candidate, sources)

    # Save to sentiment_overrides.json
    current_overrides[suggestion_id] = item_overrides
    os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(current_overrides, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return merged_candidate


def get_all_library_items_with_overrides() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return all items merged with overrides (for admin routes), along with sources."""
    raw = load_raw_library()
    sources = raw.get("sources", {})
    overrides = load_overrides()

    merged_items: List[Dict[str, Any]] = []
    for item in raw.get("suggestions", []):
        sid = item.get("id")
        if sid and sid in overrides:
            item = {**item, **overrides[sid]}
        merged_items.append(item)

    return merged_items, sources


def load_served_suggestions(
    dev_include_drafts: Optional[bool] = None,
    app_env: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return served suggestions according to status and environment rules.

    Rules:
    - Status 'approved' is always served.
    - Status 'draft' is served ONLY IF DEV_INCLUDE_DRAFTS=true AND APP_ENV=development.
    - In any other APP_ENV, draft items are ignored.
    """
    env = (app_env if app_env is not None else APP_ENV).strip().lower()
    include_drafts_flag = (
        DEV_INCLUDE_DRAFTS if dev_include_drafts is None else dev_include_drafts
    )
    allow_drafts = bool(include_drafts_flag and env == "development")

    items, _ = get_all_library_items_with_overrides()

    served: List[Dict[str, Any]] = []
    for item in items:
        status = item.get("status", "draft")
        if status == "approved":
            served.append(item)
        elif status == "draft" and allow_drafts:
            served.append(item)

    return served


def get_theme_labels() -> Dict[str, str]:
    """Retrieve mapping of theme -> English label for why_this text."""
    raw = load_raw_library()
    return raw.get("theme_labels_en", {})
