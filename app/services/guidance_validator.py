"""Validation functions for Guidance AI Selections and Rewording.

Rules:
- Selection: subset of candidates, no duplicates, length 1-4, no extra keys.
- Reword: max 1.3x length, no new digits, no phone numbers/URLs unless in original,
  none of forbidden words, none of medication words, correct script, no extra IDs.
- On any failure, drops the item and uses approved library text.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
FORBIDDEN_WORDS_PATH = os.path.join(CONFIG_DIR, "forbidden_words.json")
MEDICATION_WORDS_PATH = os.path.join(CONFIG_DIR, "medication_words.json")

# URL pattern
URL_REGEX = re.compile(r"https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# Phone number / helpline pattern (3 or more connected digits)
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,4}[-.\s]*)?(?:\(?\d{2,5}\)?[-.\s]*)?\d{3,5}[-.\s]*\d{3,5}\b")

# Script ranges
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
BENGALI_RANGE = re.compile(r"[\u0980-\u09FF]")
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# Non-ASCII digit mapping (Devanagari, Bengali, Tamil digits)
ALL_DIGITS_REGEX = re.compile(r"[\d\u0966-\u096F\u09E6-\u09EF\u0BE6-\u0BEF]")


def _load_forbidden_words() -> Dict[str, List[str]]:
    if not os.path.exists(FORBIDDEN_WORDS_PATH):
        return {}
    with open(FORBIDDEN_WORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_medication_words() -> Dict[str, List[str]]:
    if not os.path.exists(MEDICATION_WORDS_PATH):
        return {}
    with open(MEDICATION_WORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}


def validate_selection_payload(
    raw_response: Any,
    candidate_ids: List[str],
) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """Validate model selection response.

    Rules:
    - Must be a dictionary.
    - Only allowed keys: 'selected_ids', 'reason_tags'.
    - selected_ids must be a non-empty list of length 1 to 4.
    - Every ID must exist in candidate_ids.
    - No duplicates.

    Returns:
        (is_valid, selected_ids, error_type)
    """
    if not isinstance(raw_response, dict):
        return False, None, "not_a_dict"

    allowed_keys = {"selected_ids", "reason_tags"}
    extra_keys = set(raw_response.keys()) - allowed_keys
    if extra_keys:
        return False, None, "extra_keys"

    selected_ids = raw_response.get("selected_ids")
    if not isinstance(selected_ids, list):
        return False, None, "selected_ids_not_a_list"

    if len(selected_ids) == 0:
        return False, None, "empty_selection"

    if len(selected_ids) > 4:
        return False, None, "too_many_ids"

    if len(selected_ids) != len(set(selected_ids)):
        return False, None, "duplicate_ids"

    candidate_set = set(candidate_ids)
    for sid in selected_ids:
        if not isinstance(sid, str) or sid not in candidate_set:
            return False, None, "invalid_id"

    return True, selected_ids, None


def validate_reword_text(
    rewritten_text: str,
    original_text: str,
    language: str,
) -> Tuple[bool, Optional[str]]:
    """Validate a single reworded sentence against strict safety checks.

    Rules:
    - Max 1.3x original word count.
    - No new digits (only digits existing in original).
    - No URLs or phone numbers unless existing in original.
    - No forbidden diagnostic words.
    - No medication words.
    - Language script check.

    Returns:
        (is_valid, failure_type)
    """
    if not rewritten_text or not isinstance(rewritten_text, str):
        return False, "empty_text"

    clean_rewrite = rewritten_text.strip()
    clean_orig = (original_text or "").strip()

    # 1. Length check: at most 1.3 times original word count
    orig_words = len(clean_orig.split())
    max_words = max(int(orig_words * 1.3 + 0.999), orig_words + 2)
    rewrite_words = len(clean_rewrite.split())
    if rewrite_words > max_words:
        return False, "reword_length"

    # 2. URLs check
    orig_urls = set(URL_REGEX.findall(clean_orig))
    rewrite_urls = set(URL_REGEX.findall(clean_rewrite))
    if rewrite_urls - orig_urls:
        return False, "reword_url"

    # 3. Phone numbers / helplines check
    orig_phones = set(PHONE_REGEX.findall(clean_orig))
    rewrite_phones = set(PHONE_REGEX.findall(clean_rewrite))
    if rewrite_phones - orig_phones:
        return False, "reword_phone"

    # 4. Digits check: no digits unless the exact same digits are in the original
    orig_digits = set(ALL_DIGITS_REGEX.findall(clean_orig))
    rewrite_digits = set(ALL_DIGITS_REGEX.findall(clean_rewrite))
    new_digits = rewrite_digits - orig_digits
    if new_digits:
        return False, "reword_digit"

    # 5. Forbidden words check
    forbidden_dict = _load_forbidden_words()
    rewrite_lower = clean_rewrite.lower()
    # Check English + target language lists
    check_terms = list(forbidden_dict.get("en", []))
    if language != "en":
        check_terms.extend(forbidden_dict.get(language, []))

    for term in check_terms:
        if term.lower() in rewrite_lower:
            return False, "reword_forbidden_word"

    # 6. Medication words check
    med_dict = _load_medication_words()
    # Check all medication lists across languages
    for lang_code, words in med_dict.items():
        for term in words:
            if term.lower() in rewrite_lower:
                return False, "reword_medication_word"

    # 7. Script check
    if language == "hi":
        if not DEVANAGARI_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
        if LATIN_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
    elif language == "bn":
        if not BENGALI_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
        if LATIN_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
    elif language == "ta":
        if not TAMIL_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
        if LATIN_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
    elif language == "en":
        if not LATIN_RANGE.search(clean_rewrite):
            return False, "reword_script_mismatch"
        # Disallow Indic scripts in English
        if (
            DEVANAGARI_RANGE.search(clean_rewrite)
            or BENGALI_RANGE.search(clean_rewrite)
            or TAMIL_RANGE.search(clean_rewrite)
        ):
            return False, "reword_script_mismatch"

    return True, None
