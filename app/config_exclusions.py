"""Configuration for translation key exclusions.

Certain sensitive domains (such as mental health clinical questionnaires
and crisis/safety prompts) must NEVER be machine-translated.
"""

from typing import List

# Keys starting with these prefixes must never be machine translated
EXCLUDED_KEY_PREFIXES: List[str] = [
    "questionnaire.",
    "safety.",
]


def is_excluded_key(key: str) -> bool:
    """Check if a translation key starts with any forbidden prefix."""
    return any(key.startswith(prefix) for prefix in EXCLUDED_KEY_PREFIXES)
