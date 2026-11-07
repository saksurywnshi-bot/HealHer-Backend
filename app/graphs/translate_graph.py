"""LangGraph Translation Workflow.

Workflow for translating HealHer application content into Hindi, Bengali, and Tamil.
Designed to be modular so future workflows (e.g. maternal insights, risk sentiment)
can be attached as separate graphs.

State:
- language: str ('hi', 'bn', 'ta')
- items: list of dicts [{'key': str, 'text': str}]
- results: dict of key -> {'text': str, 'status': 'reviewed' | 'machine' | 'fallback'}
- missing_items: list of items needing translation
- batch_translations: candidate translations returned from LLM
- attempt: int (1 or 2)
- errors: list of error strings
- final_response: dict
- custom_llm_fn: optional async callable for testing/mocking
"""

import re
from typing import TypedDict, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, START, END

from app.config import SUPPORTED_LANGUAGES, MAX_BATCH_SIZE
from app.config_exclusions import is_excluded_key
from app.services.cache import cache_service, compute_source_hash
from app.services.llm import translate_batch_with_llm

PLACEHOLDER_REGEX = re.compile(
    r"(\{\{[^}]+\}\}|\{[a-zA-Z0-9_]+\}|%(?:\([a-zA-Z0-9_]+\))?[0-9]*[sdif]|%s)"
)


def extract_placeholders(text: str) -> List[str]:
    """Extract all format and template placeholders from text."""
    return PLACEHOLDER_REGEX.findall(text)


class TranslateState(TypedDict, total=False):
    language: str
    items: List[Dict[str, str]]
    results: Dict[str, Dict[str, str]]
    missing_items: List[Dict[str, str]]
    batch_translations: Dict[str, str]
    attempt: int
    errors: List[str]
    final_response: Dict[str, Any]
    custom_llm_fn: Optional[Callable]


# Node 1: validate_input
def validate_input(state: TranslateState) -> Dict[str, Any]:
    """Check language in hi/bn/ta, max 100 items, non-empty text."""
    errors: List[str] = list(state.get("errors", []))
    language = state.get("language", "")
    items = state.get("items", [])

    if language not in SUPPORTED_LANGUAGES:
        errors.append(
            f"Unsupported language '{language}'. Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )

    if not items:
        errors.append("Items list cannot be empty.")
    elif len(items) > MAX_BATCH_SIZE:
        errors.append(f"Maximum items per batch is {MAX_BATCH_SIZE}. Received {len(items)}.")

    for i, item in enumerate(items):
        key = item.get("key", "").strip() if isinstance(item, dict) else ""
        text = item.get("text", "").strip() if isinstance(item, dict) else ""
        if not key:
            errors.append(f"Item at index {i} has empty or missing key.")
        if not text:
            errors.append(f"Item at index {i} (key: '{key}') has empty text.")

    return {
        "errors": errors,
        "results": dict(state.get("results", {})),
        "missing_items": [],
        "attempt": state.get("attempt", 1),
    }


# Node 2: check_exclusions
def check_exclusions(state: TranslateState) -> Dict[str, Any]:
    """Reject any key starting with 'questionnaire.' or 'safety.'."""
    errors: List[str] = list(state.get("errors", []))
    items = state.get("items", [])

    for item in items:
        key = item.get("key", "")
        if is_excluded_key(key):
            errors.append(
                f"Key '{key}' is excluded from machine translation for safety/clinical reasons."
            )

    return {"errors": errors}


def route_after_exclusions(state: TranslateState) -> str:
    """If errors exist after validation and exclusion check, jump directly to respond."""
    if state.get("errors"):
        return "respond"
    return "load_cache"


# Node 3: load_cache
def load_cache(state: TranslateState) -> Dict[str, Any]:
    """Use 'reviewed' translation if present, else 'machine' if source hash matches.

    Put the rest in missing_items.
    """
    if state.get("errors"):
        return {}

    language = state["language"]
    items = state.get("items", [])
    results: Dict[str, Dict[str, str]] = dict(state.get("results", {}))
    missing_items: List[Dict[str, str]] = []

    keys = [item["key"] for item in items]
    cached_map = cache_service.get_translations_batch(keys, language)

    for item in items:
        key = item["key"]
        text = item["text"]
        current_hash = compute_source_hash(text)
        cached = cached_map.get(key)

        if cached:
            if cached["status"] == "reviewed":
                results[key] = {
                    "text": cached["translated_text"],
                    "status": "reviewed",
                }
            elif cached["status"] == "machine" and cached["source_hash"] == current_hash:
                results[key] = {
                    "text": cached["translated_text"],
                    "status": "machine",
                }
            else:
                missing_items.append(item)
        else:
            missing_items.append(item)

    return {"results": results, "missing_items": missing_items}


# Node 4: translate_batch
async def translate_batch(state: TranslateState) -> Dict[str, Any]:
    """Call Gemini once for all missing items using structured JSON output.

    10-second timeout enforced.
    """
    if state.get("errors") or not state.get("missing_items"):
        return {"batch_translations": {}}

    missing = state["missing_items"]
    language = state["language"]
    attempt = state.get("attempt", 1)

    custom_fn = state.get("custom_llm_fn")
    try:
        if custom_fn is not None:
            translations = await custom_fn(missing, language, attempt)
        else:
            translations = await translate_batch_with_llm(missing, language, attempt)
    except Exception as exc:
        translations = {}

    return {"batch_translations": translations}


# Node 5: validate_placeholders
def validate_placeholders(state: TranslateState) -> Dict[str, Any]:
    """Check every placeholder in the English text appears unchanged in the translation.

    Keys that fail go back to missing_items.
    """
    if state.get("errors"):
        return {}

    missing = state.get("missing_items", [])
    batch_translations = state.get("batch_translations", {})
    results = dict(state.get("results", {}))
    failed_items: List[Dict[str, str]] = []

    for item in missing:
        key = item["key"]
        source_text = item["text"]
        translated_text = batch_translations.get(key)

        if translated_text and translated_text.strip():
            src_placeholders = sorted(extract_placeholders(source_text))
            tgt_placeholders = sorted(extract_placeholders(translated_text))

            if src_placeholders == tgt_placeholders:
                # Valid translation preserving all placeholders
                results[key] = {
                    "text": translated_text,
                    "status": "machine",
                }
            else:
                # Placeholder mismatch: fails validation
                failed_items.append(item)
        else:
            # Missing or empty response from LLM
            failed_items.append(item)

    return {"results": results, "missing_items": failed_items}


# Conditional Edge: after validate_placeholders
def route_after_placeholder_validation(state: TranslateState) -> str:
    """If some keys failed validation and attempt < 2, route back to prepare_retry."""
    if state.get("errors"):
        return "respond"

    failed = state.get("missing_items", [])
    attempt = state.get("attempt", 1)

    if failed and attempt < 2:
        return "prepare_retry"
    return "fallback"


# Node 6 (helper for retry): prepare_retry
def prepare_retry(state: TranslateState) -> Dict[str, Any]:
    """Increment attempt count before retrying translation."""
    return {"attempt": state.get("attempt", 1) + 1}


# Node 7: fallback
def fallback(state: TranslateState) -> Dict[str, Any]:
    """For any key still missing, or if Gemini failed, return English text with status 'fallback'.

    Never raise errors to the client.
    """
    if state.get("errors"):
        return {}

    items = state.get("items", [])
    results = dict(state.get("results", {}))

    for item in items:
        key = item["key"]
        if key not in results:
            results[key] = {
                "text": item["text"],
                "status": "fallback",
            }

    return {"results": results, "missing_items": []}


# Node 8: save_cache
def save_cache(state: TranslateState) -> Dict[str, Any]:
    """Store new valid translations as 'machine'."""
    if state.get("errors"):
        return {}

    language = state["language"]
    items = state.get("items", [])
    results = state.get("results", {})

    items_dict = {item["key"]: item["text"] for item in items}

    for key, res in results.items():
        if res.get("status") == "machine":
            source_text = items_dict.get(key, "")
            cache_service.save_machine_translation(
                key=key,
                language=language,
                source_text=source_text,
                translated_text=res["text"],
            )

    return {}


# Node 9: respond
def respond(state: TranslateState) -> Dict[str, Any]:
    """Build the final response."""
    errors = state.get("errors", [])
    if errors:
        return {
            "final_response": {
                "language": state.get("language", ""),
                "errors": errors,
            }
        }

    return {
        "final_response": {
            "language": state["language"],
            "translations": state.get("results", {}),
        }
    }


def build_translation_graph():
    """Build and compile the translation state graph."""
    workflow = StateGraph(TranslateState)

    # Add nodes
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("check_exclusions", check_exclusions)
    workflow.add_node("load_cache", load_cache)
    workflow.add_node("translate_batch", translate_batch)
    workflow.add_node("validate_placeholders", validate_placeholders)
    workflow.add_node("prepare_retry", prepare_retry)
    workflow.add_node("fallback", fallback)
    workflow.add_node("save_cache", save_cache)
    workflow.add_node("respond", respond)

    # Edges
    workflow.add_edge(START, "validate_input")
    workflow.add_edge("validate_input", "check_exclusions")

    workflow.add_conditional_edges(
        "check_exclusions",
        route_after_exclusions,
        {
            "load_cache": "load_cache",
            "respond": "respond",
        },
    )

    workflow.add_edge("load_cache", "translate_batch")
    workflow.add_edge("translate_batch", "validate_placeholders")

    workflow.add_conditional_edges(
        "validate_placeholders",
        route_after_placeholder_validation,
        {
            "prepare_retry": "prepare_retry",
            "fallback": "fallback",
            "respond": "respond",
        },
    )

    workflow.add_edge("prepare_retry", "translate_batch")
    workflow.add_edge("fallback", "save_cache")
    workflow.add_edge("save_cache", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


# Compile once at startup
translation_graph = build_translation_graph()


def get_translation_graph_mermaid() -> str:
    """Export the translation workflow graph as a Mermaid diagram."""
    return translation_graph.get_graph().draw_mermaid()
