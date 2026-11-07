"""BEAM Perinatal Insights Configuration Loader.

Loads all clinical thresholds, tier triggers, suggestion libraries, safety phrases,
and screening questionnaires from versioned JSON files.
NO MAGIC NUMBERS IN CODE: changing a threshold requires only editing the config JSON.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
RULES_FILE = os.path.join(CONFIG_DIR, "beam_perinatal_rules.json")
SUGGESTIONS_FILE = os.path.join(CONFIG_DIR, "beam_suggestions.json")
SAFETY_PHRASES_FILE = os.path.join(CONFIG_DIR, "beam_safety_phrases.json")
QUESTIONNAIRES_FILE = os.path.join(CONFIG_DIR, "beam_questionnaires.json")


def load_json(filepath: str) -> Dict[str, Any]:
    """Safely read and parse a JSON configuration file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BEAM config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_beam_rules_config() -> Dict[str, Any]:
    """Retrieve the cached BEAM rules configuration."""
    return load_json(RULES_FILE)


@lru_cache(maxsize=1)
def get_beam_suggestions_config() -> Dict[str, Any]:
    """Retrieve the cached BEAM suggestions configuration."""
    return load_json(SUGGESTIONS_FILE)


@lru_cache(maxsize=1)
def get_beam_safety_phrases_config() -> Dict[str, Any]:
    """Retrieve the cached safety phrase lists for real-time Tier 3 detection."""
    return load_json(SAFETY_PHRASES_FILE)


@lru_cache(maxsize=1)
def get_beam_questionnaires_config() -> Dict[str, Any]:
    """Retrieve the cached screening questionnaires configuration."""
    return load_json(QUESTIONNAIRES_FILE)


def clear_config_caches() -> None:
    """Clear all in-memory config caches (useful in tests when testing custom configs)."""
    get_beam_rules_config.cache_clear()
    get_beam_suggestions_config.cache_clear()
    get_beam_safety_phrases_config.cache_clear()
    get_beam_questionnaires_config.cache_clear()
