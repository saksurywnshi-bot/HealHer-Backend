"""Robust script to translate HealHer app content via the LangGraph FastAPI backend.

Processes keys in manageable batches of 15 items with retry and timeout protection.
"""

import os
import sys
import json
import time
import shutil
import httpx
import re

BACKEND_URL = "http://localhost:3001/api/translate"
LOCALES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../lovable_heal_her/src/i18n/locales")
)

LANGUAGES = ["hi", "bn", "ta"]
BATCH_SIZE = 15


def flatten(d, prefix=""):
    items = {}
    for k, v in d.items():
        pre = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, pre))
        elif isinstance(v, str):
            items[pre] = v
    return items


def set_deep(d, path, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def needs_translation(text: str) -> bool:
    """Check if text contains letters to translate (not just numbers, symbols, placeholders)."""
    cleaned = re.sub(r"\{\{[^}]+\}\}", "", text)
    cleaned = re.sub(r"\{[a-zA-Z0-9_]+\}", "", cleaned)
    cleaned = re.sub(r"%s", "", cleaned)
    return bool(re.search(r"[A-Za-z]", cleaned))


def translate_batch_via_backend(language: str, items: list[dict], max_retries: int = 2) -> dict[str, str]:
    """Call backend /api/translate endpoint with retry protection."""
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                BACKEND_URL,
                json={"language": language, "items": items},
                timeout=60.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                translations = data.get("translations", {})
                return {
                    k: v["text"]
                    for k, v in translations.items()
                    if v.get("status") in ("machine", "reviewed")
                }
            else:
                print(f" [HTTP {resp.status_code}]", end="", flush=True)
        except Exception as e:
            print(f" [{type(e).__name__}]", end="", flush=True)
            time.sleep(2.0)
    return {}


def main():
    en_path = os.path.join(LOCALES_DIR, "en.json")
    if not os.path.exists(en_path):
        print(f"Error: {en_path} not found.")
        sys.exit(1)

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    flat_en = flatten(en_data)

    # Exclude questionnaire.* and safety.*
    eligible_keys = [
        k
        for k, v in flat_en.items()
        if not k.startswith("questionnaire.")
        and not k.startswith("safety.")
        and needs_translation(v)
    ]

    # Prioritize visible screens first: welcome, common, hub, mood, sleep, profile, options
    screen_priority = ["welcome.", "common.", "hub.", "mood.", "sleep.", "signup.", "profile.", "options.", "support.", "caregiver."]
    def key_priority(k):
        for idx, prefix in enumerate(screen_priority):
            if k.startswith(prefix):
                return idx
        return 999

    eligible_keys.sort(key=key_priority)

    print(f"Total eligible keys in app: {len(eligible_keys)}")

    for lang in LANGUAGES:
        lang_path = os.path.join(LOCALES_DIR, f"{lang}.json")
        backup_path = os.path.join(LOCALES_DIR, f"{lang}.json.bak")

        if not os.path.exists(backup_path) and os.path.exists(lang_path):
            shutil.copyfile(lang_path, backup_path)
            print(f"Backed up {lang}.json -> {lang}.json.bak")

        if os.path.exists(lang_path):
            with open(lang_path, "r", encoding="utf-8") as f:
                lang_data = json.load(f)
        else:
            lang_data = {}

        flat_lang = flatten(lang_data)

        # Keys that are missing or still identical to English
        untranslated = [
            k for k in eligible_keys
            if k not in flat_lang or flat_lang[k] == flat_en[k] or str(flat_lang[k]).endswith("_todo")
        ]

        print(f"\n==========================================")
        print(f"Language: {lang.upper()} — {len(untranslated)} keys to translate")
        print(f"==========================================")

        total_batches = (len(untranslated) + BATCH_SIZE - 1) // BATCH_SIZE
        success_count = 0

        for i in range(0, len(untranslated), BATCH_SIZE):
            batch_keys = untranslated[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            items = [{"key": k, "text": flat_en[k]} for k in batch_keys]

            print(f"[{lang}] Batch {batch_num}/{total_batches} ({len(items)} items)... ", end="", flush=True)
            results = translate_batch_via_backend(lang, items)

            if results:
                for k, text in results.items():
                    set_deep(lang_data, k, text)
                success_count += len(results)
                print(f"Done ({len(results)} translated)")
            else:
                print("Skipped")

            # Save progress after every batch
            with open(lang_path, "w", encoding="utf-8") as f:
                json.dump(lang_data, f, ensure_ascii=False, indent=2)

            time.sleep(0.5)

        print(f"Finished {lang.upper()}! Total translated: {success_count}/{len(untranslated)}")


if __name__ == "__main__":
    main()
