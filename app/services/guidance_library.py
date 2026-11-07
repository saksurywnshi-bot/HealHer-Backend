"""Library service for Questionnaire Suggestions.

Rules:
- data/questionnaire_suggestions.json + data/library_overrides.json is the ONLY source.
- Model may only CHOOSE and ORDER approved library items, and optionally REWORD them.
- It may never add advice, facts, numbers or medical content.
- Changes to either file change the library_version hash and invalidate the cache.
- "mixed" evidence items (e.g. calming practice) never appear in moderate or higher bands.
- First message in see_clinician or higher is always the clinician-contact message.
- Only verified helplines are returned (never 1800-889-4357, 102, 108).
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set

from app.config import DEV_INCLUDE_DRAFTS

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

SUGGESTIONS_PATH = os.path.join(DATA_DIR, "questionnaire_suggestions.json")
OVERRIDES_PATH = os.path.join(DATA_DIR, "library_overrides.json")
THEME_MAP_PATH = os.path.join(CONFIG_DIR, "theme_to_suggestions.json")
FORBIDDEN_WORDS_PATH = os.path.join(CONFIG_DIR, "forbidden_words.json")
MEDICATION_WORDS_PATH = os.path.join(CONFIG_DIR, "medication_words.json")

# Verified India Helplines (Strict rule: never contain 1800-889-4357, 102, or 108)
VERIFIED_HELPLINES: Dict[str, str] = {
    "tele_manas": "14416",
    "emergency": "112",
    "women_helpline": "181",
}

MODERATE_OR_HIGHER_BANDS: Set[str] = {
    "moderate",
    "moderately_severe",
    "severe",
    "possible_depression",
}


def get_library_version_hash() -> str:
    """Compute SHA-256 hash of suggestions library and overrides.

    Any edit to either file changes the hash and invalidates cached guidance.
    """
    hasher = hashlib.sha256()
    if os.path.exists(SUGGESTIONS_PATH):
        with open(SUGGESTIONS_PATH, "rb") as f:
            hasher.update(f.read())
    else:
        hasher.update(b"[]")

    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, "rb") as f:
            hasher.update(f.read())
    else:
        hasher.update(b"{}")

    return hasher.hexdigest()[:16]


def load_theme_map() -> Dict[str, List[str]]:
    """Load theme to suggestion IDs mapping."""
    if not os.path.exists(THEME_MAP_PATH):
        return {}
    with open(THEME_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}


def load_library(dev_include_drafts: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Load questionnaire suggestions merged with overrides."""
    include_drafts = (
        DEV_INCLUDE_DRAFTS if dev_include_drafts is None else dev_include_drafts
    )
    if not os.path.exists(SUGGESTIONS_PATH):
        return []

    with open(SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    overrides = {}
    if os.path.exists(OVERRIDES_PATH):
        try:
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except Exception:
            overrides = {}

    merged: List[Dict[str, Any]] = []
    for item in items:
        item_id = item.get("id", "")
        if item_id in overrides:
            item = {**item, **overrides[item_id]}

        status = item.get("status", "draft")
        if status != "approved" and not include_drafts:
            continue

        merged.append(item)

    return merged


def build_candidates(
    band_id: str,
    stage: str,
    themes: Optional[List[str]] = None,
    dev_include_drafts: Optional[bool] = None,
    escalation: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter and rank candidate suggestions for the given band, stage, and themes.

    Constraints:
    - Status: approved (or draft in dev with DEV_INCLUDE_DRAFTS).
    - show_only_in_bands: must match band_id if specified (unless clinician escalation warrants clinician contact).
    - applies_to_stages: must match stage if specified.
    - Mixed evidence items NEVER appear in moderate or higher bands (or clinician escalation).
    - Tag candidates matching theme_to_suggestions.json and rank them first.
    - Maximum 8 candidates returned.
    """
    library = load_library(dev_include_drafts=dev_include_drafts)
    theme_map = load_theme_map()
    norm_themes = set(t.lower() for t in (themes or []))

    # Determine theme-tagged suggestion IDs
    theme_tagged_ids: Set[str] = set()
    for t in norm_themes:
        for sug_id in theme_map.get(t, []):
            theme_tagged_ids.add(sug_id)

    candidates: List[Dict[str, Any]] = []
    is_clinician_escalation = escalation in ("see_clinician", "see_clinician_soon", "see_clinician_urgent")
    is_moderate_or_higher = (band_id.lower() in MODERATE_OR_HIGHER_BANDS) or is_clinician_escalation

    for item in library:
        # Check band allowance
        allowed_bands = item.get("show_only_in_bands")
        if allowed_bands and band_id.lower() not in [b.lower() for b in allowed_bands]:
            # Always permit clinician contact if clinical escalation is active
            if not (is_clinician_escalation and (item.get("is_clinician_contact") or item.get("id") == "SUG_CLINICIAN_CONTACT")):
                continue

        # Check stage allowance
        allowed_stages = item.get("applies_to_stages")
        if allowed_stages and stage.lower() not in [s.lower() for s in allowed_stages] and "any" not in allowed_stages:
            continue

        # Strict evidence rule: "mixed" evidence never in moderate or higher band
        evidence = str(item.get("evidence", "")).lower()
        if is_moderate_or_higher and evidence == "mixed":
            continue

        matched_theme_count = 1 if item.get("id") in theme_tagged_ids else 0
        candidates.append({
            "item": item,
            "theme_match": matched_theme_count,
        })

    # Sort candidates: theme-matched candidates first, maintaining library order
    candidates.sort(key=lambda c: c["theme_match"], reverse=True)

    # Return up to 8 candidate items
    return [c["item"] for c in candidates[:8]]


def order_deterministic(
    candidates: List[Dict[str, Any]],
    escalation: str,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """Deterministic selection of up to 4 suggestions.

    Rule: If escalation is see_clinician or higher, the clinician-contact
    message MUST stay first.
    """
    if not candidates:
        return []

    # Separate clinician contact and regular items
    clinician_items = [c for c in candidates if c.get("is_clinician_contact") or c.get("id") == "SUG_CLINICIAN_CONTACT"]
    other_items = [c for c in candidates if not (c.get("is_clinician_contact") or c.get("id") == "SUG_CLINICIAN_CONTACT")]

    ordered_raw: List[Dict[str, Any]] = []
    if escalation in ("see_clinician", "see_clinician_soon", "see_clinician_urgent") and clinician_items:
        ordered_raw.append(clinician_items[0])
        ordered_raw.extend(other_items[:3])
    else:
        ordered_raw = candidates[:4]

    formatted: List[Dict[str, Any]] = []
    for item in ordered_raw:
        text_dict = item.get("text", {})
        text = text_dict.get(language) or text_dict.get("en", "")
        formatted.append({
            "id": item.get("id", ""),
            "category": item.get("category", ""),
            "text": text,
        })

    return formatted


def get_safe_fallback_text(language: str = "en") -> str:
    """Safe baseline guidance text."""
    fallbacks = {
        "en": "Take things one day at a time, rest where you can, and connect with people who support you. Free mental health support is available anytime through Tele-MANAS (14416) or emergency services (112).",
        "hi": "चीजों को एक दिन में एक बार लें, जहां संभव हो आराम करें और उन लोगों से जुड़ें जो आपका समर्थन करते हैं। टेली-मानस (14416) या आपातकालीन सेवाओं (112) के माध्यम से किसी भी समय मुफ्त सहायता उपलब्ध है।",
        "bn": "প্রতিটি দিনকে সহজভাবে নিন, যেখানে সম্ভব বিশ্রাম নিন এবং সহায়তাকারী ব্যক্তিদের সাথে যোগাযোগ রাখুন। টেলি-মানস (14416) বা জরুরি সেবার (112) মাধ্যমে যেকোনো সময় বিনামূল্যে সহায়তা পাওয়া যায়।",
        "ta": "ஒவ்வொரு நாளையும் நிதானமாக அணுகுங்கள், முடிந்தவரை ஓய்வெடுங்கள், ஆதரவளிக்கும் நபர்களுடன் தொடர்பில் இருங்கள். டெலி-மானாஸ் (14416) அல்லது அவசர சேவைகள் (112) மூலம் இலவச உதவி எப்போதும் கிடைக்கும்.",
    }
    return fallbacks.get(language, fallbacks["en"])


def get_verified_helplines() -> Dict[str, str]:
    """Return strictly verified helplines (never 1800-889-4357, 102, 108)."""
    return dict(VERIFIED_HELPLINES)
