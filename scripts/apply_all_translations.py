"""Master script to apply all translations (parts 1 to 5) to:
1. lovable_heal_her/src/i18n/locales/hi.json
2. lovable_heal_her/src/i18n/locales/bn.json
3. lovable_heal_her/src/i18n/locales/ta.json
4. Backend_heal_her/translations.db (SQLite cache)

Leaves clinical questionnaires (EPDS, PHQ-9, Mother-Child Bonding PBQ, Family observations)
and safety helplines untranslated / as-is per user instructions.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import sys

# Import all 5 translation parts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import translations_part1
import translations_part2
import translations_part3
import translations_part4
import translations_part5


def get_all_translations():
    all_data = {}
    for mod in [
        translations_part1,
        translations_part2,
        translations_part3,
        translations_part4,
        translations_part5,
    ]:
        all_data.update(mod.DATA)
    return all_data


def set_deep(target, path, value):
    parts = path.split(".")
    cur = target
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def flatten(d, prefix=""):
    items = {}
    for k, v in d.items():
        new_k = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, new_k))
        else:
            items[new_k] = v
    return items


def main():
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    locales_dir = os.path.join(root, "lovable_heal_her", "src", "i18n", "locales")
    db_path = os.path.join(root, "Backend_heal_her", "translations.db")

    en_path = os.path.join(locales_dir, "en.json")
    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    flat_en = flatten(en_data)
    translations = get_all_translations()

    print(f"Loaded {len(translations)} total translated keys across parts 1–5.")

    # SQLite connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            key TEXT NOT NULL,
            language TEXT NOT NULL,
            source_text TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (key, language)
        )
    """)

    now_iso = datetime.now(timezone.utc).isoformat()

    languages = ["hi", "bn", "ta"]

    for lang in languages:
        target_path = os.path.join(locales_dir, f"{lang}.json")
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                lang_data = json.load(f)
        else:
            lang_data = copy.deepcopy(en_data)

        applied_count = 0
        db_count = 0

        for key, lang_map in translations.items():
            if lang in lang_map:
                val = lang_map[lang]
                set_deep(lang_data, key, val)
                applied_count += 1

                # Sync to SQLite
                src = flat_en.get(key, "")
                if isinstance(src, (dict, list)):
                    src_str = json.dumps(src, ensure_ascii=False)
                else:
                    src_str = str(src)

                if isinstance(val, (dict, list)):
                    val_str = json.dumps(val, ensure_ascii=False)
                else:
                    val_str = str(val)

                src_hash = hashlib.sha256(src_str.encode("utf-8")).hexdigest()

                cursor.execute(
                    """
                    INSERT INTO translations (key, language, source_text, source_hash, translated_text, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'reviewed', ?)
                    ON CONFLICT(key, language) DO UPDATE SET
                        source_text=excluded.source_text,
                        source_hash=excluded.source_hash,
                        translated_text=excluded.translated_text,
                        status='reviewed',
                        updated_at=excluded.updated_at
                """,
                    (key, lang, src_str, src_hash, val_str, now_iso),
                )
                db_count += 1

        # Write out updated locale json
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(lang_data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(
            f"[{lang.upper()}] Applied {applied_count} translations to {target_path} and synced {db_count} to SQLite."
        )

    conn.commit()
    conn.close()

    # Final verification of translated vs untranslated
    print("\n--- TRANSLATION VERIFICATION ---")
    for lang in languages:
        target_path = os.path.join(locales_dir, f"{lang}.json")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        flat_lang = flatten(data)

        translated = 0
        identical_to_en = 0
        excluded_count = 0

        for k, v in flat_lang.items():
            prefix = k.split(".")[0]
            if prefix in ["questionnaires", "safety"]:
                excluded_count += 1
                continue
            en_val = flat_en.get(k)
            if v == en_val:
                identical_to_en += 1
            else:
                translated += 1

        print(
            f"{lang.upper()}: {translated} app screen keys translated | {identical_to_en} identical to EN | {excluded_count} questionnaires/safety kept in EN"
        )


if __name__ == "__main__":
    main()
