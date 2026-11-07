"""Caregiver Daily Tips Library Loader & Clinical Validator.

Validates:
- Every source id in source_refs exists in sources dictionary.
- Every theme, kind, effort, and stage matches the allowed lists in the file.
- Approved tips must have needs_source == False and non-empty source_refs.
- text_en is non-empty, contains NO digits (0-9), and contains NO forbidden non-diagnostic
  or medication words (from config/forbidden_words.json and config/medication_words.json).
- Merges clinician overrides from data/caregiver_overrides.json.
- Serves only status 'approved', unless DEV_INCLUDE_DRAFTS=true AND APP_ENV=development.
- Exposes a deterministic SHA-256 library version hash.
"""

import copy
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import app.config as config

logger = logging.getLogger("healher_backend.caregiver_library")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY_FILE_PATH = os.path.join(_BASE_DIR, "data", "caregiver_suggestions.json")
OVERRIDES_FILE_PATH = os.path.join(_BASE_DIR, "data", "caregiver_overrides.json")
FORBIDDEN_WORDS_PATH = os.path.join(_BASE_DIR, "app", "config", "forbidden_words.json")
MEDICATION_WORDS_PATH = os.path.join(_BASE_DIR, "app", "config", "medication_words.json")

# In-memory cached validated library state
_CACHED_RAW_LIBRARY: Optional[Dict[str, Any]] = None
_CACHED_SERVED_LIBRARY: Optional[Dict[str, Any]] = None
_CACHED_VERSION_HASH: Optional[str] = None


def load_forbidden_words() -> Set[str]:
    """Load forbidden words and medication words across all supported languages."""
    words: Set[str] = {
        "mild", "moderate", "severe", "depressed", "depression",
        "diagnosis", "diagnos", "diagnose", "disorder", "clinical",
        "bipolar", "psychosis", "schizo", "syndrome", "pathology", "mental illness"
    }
    if os.path.exists(FORBIDDEN_WORDS_PATH):
        try:
            with open(FORBIDDEN_WORDS_PATH, "r", encoding="utf-8") as f:
                fb = json.load(f)
                for lang_words in fb.values():
                    if isinstance(lang_words, list):
                        for w in lang_words:
                            words.add(w.strip().lower())
        except Exception as e:
            logger.warning("Could not read forbidden_words.json: %s", e)

    if os.path.exists(MEDICATION_WORDS_PATH):
        try:
            with open(MEDICATION_WORDS_PATH, "r", encoding="utf-8") as f:
                mw = json.load(f)
                for lang_words in mw.values():
                    if isinstance(lang_words, list):
                        for w in lang_words:
                            words.add(w.strip().lower())
        except Exception as e:
            logger.warning("Could not read medication_words.json: %s", e)

    return words


def validate_caregiver_library(raw_data: Dict[str, Any]) -> None:
    """Strictly validate caregiver library data against clinical and schema constraints.
    
    Raises:
        ValueError: If any validation rule fails.
    """
    if not isinstance(raw_data, dict):
        raise ValueError("Caregiver library must be a JSON dictionary.")

    themes: List[str] = raw_data.get("themes", [])
    kinds: List[str] = raw_data.get("kinds", [])
    stages: List[str] = raw_data.get("stages", [])
    efforts: List[str] = raw_data.get("efforts", [])
    sources: Dict[str, Any] = raw_data.get("sources", {})
    suggestions: List[Dict[str, Any]] = raw_data.get("suggestions", [])

    if not themes or not kinds or not stages or not suggestions:
        raise ValueError("Caregiver library is missing required top-level lists.")

    allowed_themes = set(themes)
    allowed_kinds = set(kinds)
    allowed_stages = set(stages) | {"both"}
    allowed_efforts = set(efforts)
    allowed_sources = set(sources.keys())

    forbidden_words = load_forbidden_words()

    for idx, tip in enumerate(suggestions):
        tip_id = tip.get("id", f"tip_at_index_{idx}")

        # 1. Sources check
        source_refs = tip.get("source_refs", [])
        if not isinstance(source_refs, list):
            raise ValueError(f"Tip {tip_id}: source_refs must be a list.")
        for s in source_refs:
            if s not in allowed_sources:
                raise ValueError(
                    f"Tip {tip_id}: source_ref '{s}' is missing from sources dictionary."
                )

        # 2. Stage, kind, effort, themes check
        stage = tip.get("stage")
        if stage not in allowed_stages:
            raise ValueError(
                f"Tip {tip_id}: stage '{stage}' is not in allowed stages {allowed_stages}."
            )

        kind = tip.get("kind")
        if kind not in allowed_kinds:
            raise ValueError(
                f"Tip {tip_id}: kind '{kind}' is not in allowed kinds {allowed_kinds}."
            )

        effort = tip.get("effort")
        if effort not in allowed_efforts:
            raise ValueError(
                f"Tip {tip_id}: effort '{effort}' is not in allowed efforts {allowed_efforts}."
            )

        tip_themes = tip.get("themes", [])
        if not tip_themes or not isinstance(tip_themes, list):
            raise ValueError(f"Tip {tip_id}: themes must be a non-empty list.")
        for t in tip_themes:
            if t not in allowed_themes:
                raise ValueError(
                    f"Tip {tip_id}: theme '{t}' is not in allowed themes {allowed_themes}."
                )

        # 3. Approved status check: needs_source must be false and source_refs non-empty
        status = tip.get("status")
        needs_source = tip.get("needs_source", False)
        if status == "approved":
            if needs_source:
                raise ValueError(
                    f"Tip {tip_id}: status is 'approved' but needs_source is true."
                )
            if not source_refs:
                raise ValueError(
                    f"Tip {tip_id}: status is 'approved' but source_refs is empty."
                )

        # 4. text_en checks: non-empty, no digits, no forbidden or medication words
        text_en = tip.get("text_en")
        if not text_en or not isinstance(text_en, str) or not text_en.strip():
            raise ValueError(f"Tip {tip_id}: text_en is empty or missing.")

        # Check for digits 0-9
        if re.search(r"\d", text_en):
            raise ValueError(
                f"Tip {tip_id}: text_en contains digit(s), which is forbidden in caregiver tips: '{text_en}'"
            )

        # Check for forbidden words & medication words across text_en
        text_en_lower = text_en.lower()
        for forbidden in forbidden_words:
            # Word boundary check
            pattern = rf"\b{re.escape(forbidden)}\b"
            if re.search(pattern, text_en_lower):
                raise ValueError(
                    f"Tip {tip_id}: text_en contains forbidden or medication word '{forbidden}': '{text_en}'"
                )


def load_overrides() -> Dict[str, Dict[str, Any]]:
    """Load clinician overrides from data/caregiver_overrides.json."""
    if not os.path.exists(OVERRIDES_FILE_PATH):
        return {}
    try:
        with open(OVERRIDES_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read caregiver_overrides.json: %s", e)
        return {}


def apply_overrides(
    library: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge clinician overrides into a cloned copy of the library without altering original."""
    cloned = copy.deepcopy(library)
    if not overrides:
        return cloned

    suggestions = cloned.get("suggestions", [])
    allowed_override_fields = {"status", "text_hi", "text_bn", "text_ta", "text_en"}

    for tip in suggestions:
        tip_id = tip.get("id")
        if tip_id in overrides:
            patch = overrides[tip_id]
            for field, val in patch.items():
                if field in allowed_override_fields:
                    tip[field] = val

    return cloned


def compute_version_hash(raw_text: str, overrides_text: str) -> str:
    """Compute SHA-256 version hash combining raw library content and overrides."""
    hasher = hashlib.sha256()
    hasher.update(raw_text.encode("utf-8"))
    hasher.update(overrides_text.encode("utf-8"))
    return hasher.hexdigest()[:16]


def reload_caregiver_library() -> Dict[str, Any]:
    """Read data/caregiver_suggestions.json, validate, merge overrides, and cache.
    
    Raises:
        ValueError: If validation fails.
    """
    global _CACHED_RAW_LIBRARY, _CACHED_SERVED_LIBRARY, _CACHED_VERSION_HASH

    if not os.path.exists(LIBRARY_FILE_PATH):
        raise FileNotFoundError(f"Caregiver library file not found at {LIBRARY_FILE_PATH}")

    with open(LIBRARY_FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_data = json.loads(raw_text)

    # 1. Validate raw base file
    validate_caregiver_library(raw_data)

    # 2. Merge overrides
    overrides_text = ""
    overrides = {}
    if os.path.exists(OVERRIDES_FILE_PATH):
        with open(OVERRIDES_FILE_PATH, "r", encoding="utf-8") as f:
            overrides_text = f.read()
        try:
            overrides = json.loads(overrides_text) if overrides_text.strip() else {}
        except Exception:
            overrides = {}

    merged_data = apply_overrides(raw_data, overrides)

    # 3. Re-validate after overrides
    validate_caregiver_library(merged_data)

    # 4. Filter served tips based on APP_ENV and DEV_INCLUDE_DRAFTS
    served_suggestions = []
    # DEV_INCLUDE_DRAFTS is honored only when APP_ENV == "development"
    include_drafts = bool(config.DEV_INCLUDE_DRAFTS) and (str(config.APP_ENV).lower() == "development")

    for tip in merged_data.get("suggestions", []):
        st = tip.get("status")
        if st == "approved":
            served_suggestions.append(copy.deepcopy(tip))
        elif st == "draft" and include_drafts:
            item_copy = copy.deepcopy(tip)
            item_copy["_serve_draft"] = True
            served_suggestions.append(item_copy)

    served_lib = copy.deepcopy(merged_data)
    served_lib["suggestions"] = served_suggestions

    _CACHED_RAW_LIBRARY = merged_data
    _CACHED_SERVED_LIBRARY = served_lib
    _CACHED_VERSION_HASH = compute_version_hash(raw_text, overrides_text)

    logger.info(
        "Caregiver library loaded successfully: %d total tips, %d served (env=%s, include_drafts=%s, hash=%s)",
        len(merged_data.get("suggestions", [])),
        len(served_suggestions),
        config.APP_ENV,
        include_drafts,
        _CACHED_VERSION_HASH,
    )
    return _CACHED_SERVED_LIBRARY


load_caregiver_library = reload_caregiver_library


def get_served_caregiver_library(stage: Optional[str] = None) -> Any:
    """Return the served library dictionary (load on demand if not cached).
    If stage is provided, returns the list of suggestions for that stage.
    """
    global _CACHED_SERVED_LIBRARY
    if _CACHED_SERVED_LIBRARY is None:
        reload_caregiver_library()
    if stage is not None:
        return [
            t for t in (_CACHED_SERVED_LIBRARY or {}).get("suggestions", [])
            if t.get("stage") in (stage, "both")
        ]
    return _CACHED_SERVED_LIBRARY


def get_raw_caregiver_library() -> Dict[str, Any]:
    """Return the full library dictionary including drafts and metadata."""
    global _CACHED_RAW_LIBRARY
    if _CACHED_RAW_LIBRARY is None:
        reload_caregiver_library()
    return _CACHED_RAW_LIBRARY  # type: ignore


def get_caregiver_library_version_hash() -> str:
    """Return the SHA-256 version hash for the caregiver library."""
    global _CACHED_VERSION_HASH
    if _CACHED_VERSION_HASH is None:
        reload_caregiver_library()
    return _CACHED_VERSION_HASH or "caregiver_v0.2"
