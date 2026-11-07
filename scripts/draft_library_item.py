#!/usr/bin/env python3
"""Offline Library Drafting Tool (Admin / Clinician Preparation Only).

Usage:
    python scripts/draft_library_item.py --source-file <path_to_txt> --source-id <source_id>
    or:
    python scripts/draft_library_item.py <path_to_txt> <source_id>

Rules:
- Refuses to run if no source text is provided.
- Status is ALWAYS 'draft'.
- NEVER edits data/questionnaire_suggestions.json.
- Writes to data/drafts/<id>.json.
- Prints: "A clinician must review and approve this before use."
- No web search or external network calls other than the model API.
"""

import argparse
import json
import os
import re
import sys

# Ensure backend root is in sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv

# Load environment
env_path = os.path.join(BACKEND_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
load_dotenv()

from app.config import GEMINI_MODEL


def draft_suggestion_item(source_text: str, source_id: str) -> dict:
    """Ask Gemini model to draft one suggestion entry in library JSON format."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    system_prompt = (
        "You are an evidence-based clinical psychiatry drafting assistant for perinatal mental health. "
        "Read the provided guideline excerpt or study abstract and draft ONE new self-care suggestion entry. "
        "Strict rules: "
        "1. English text MUST be under 30 words, warm, clear, actionable, and non-diagnostic. "
        "2. Do not use forbidden diagnostic terms (depression, depressed, disorder, mild, moderate, severe, diagnosis). "
        "3. Never mention medications. "
        "4. Return ONLY valid JSON with keys: "
        "id (e.g. SUG_REST_HYGIENE), category (e.g. sleep, support, movement, coping, self_care), "
        "text (dict with key 'en' containing the suggestion under 30 words), "
        "evidence (one of: guideline, trial, expert, mixed), "
        "evidence_note (one sentence quoting or summarizing what the source says), "
        "source_refs (list of strings including the provided source_id), "
        "status (strictly 'draft'). "
        "Do not include any other markdown or text outside JSON."
    )

    user_prompt = f"Source ID: {source_id}\n\nSource Excerpt/Abstract:\n{source_text}"

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        request_timeout=20.0,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = model.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    clean_content = content.strip()
    if clean_content.startswith("```"):
        clean_content = clean_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    data = json.loads(clean_content)

    # Force status to always be 'draft'
    data["status"] = "draft"
    if "source_refs" not in data or not data["source_refs"]:
        data["source_refs"] = [source_id]
    elif source_id not in data["source_refs"]:
        data["source_refs"].append(source_id)

    # Ensure id format
    raw_id = data.get("id", "SUG_NEW_DRAFT")
    clean_id = re.sub(r"[^A-Za-z0-9_]", "", raw_id).upper()
    if not clean_id.startswith("SUG_"):
        clean_id = f"SUG_{clean_id}"
    data["id"] = clean_id

    return data


def main():
    parser = argparse.ArgumentParser(description="Draft one suggestion item from clinical abstract/guideline.")
    parser.add_argument("pos_source_file", nargs="?", default=None, help="Path to source text file.")
    parser.add_argument("pos_source_id", nargs="?", default=None, help="Source identifier (e.g. NICE_2024).")
    parser.add_argument("--source-file", "-f", dest="flag_source_file", default=None, help="Path to source text file.")
    parser.add_argument("--source-id", "-s", dest="flag_source_id", default=None, help="Source identifier.")

    args = parser.parse_args()

    source_file = args.flag_source_file or args.pos_source_file
    source_id = args.flag_source_id or args.pos_source_id

    if not source_file or not source_id:
        print("Error: Both source file and source ID are required.", file=sys.stderr)
        print("Usage: python scripts/draft_library_item.py --source-file <file.txt> --source-id <id>", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(source_file, "r", encoding="utf-8") as f:
        source_text = f.read().strip()

    if not source_text:
        print("Error: No source text provided (source file is empty).", file=sys.stderr)
        sys.exit(1)

    try:
        draft_item = draft_suggestion_item(source_text, source_id)
    except Exception as e:
        print(f"Error during draft generation: {e}", file=sys.stderr)
        sys.exit(1)

    # Save to data/drafts/<id>.json
    drafts_dir = os.path.join(BACKEND_DIR, "data", "drafts")
    os.makedirs(drafts_dir, exist_ok=True)

    item_id = draft_item.get("id", "SUG_DRAFT")
    out_path = os.path.join(drafts_dir, f"{item_id}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft_item, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Draft saved to {out_path}")
    print("A clinician must review and approve this before use.")


if __name__ == "__main__":
    main()
