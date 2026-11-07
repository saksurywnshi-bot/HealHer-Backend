HEALHER QUESTIONNAIRE GUIDANCE AI PERSONALISATION - SYSTEM GUIDE
================================================================

1. OVERVIEW
-----------
HealHer provides an optional, clinically constrained AI personalisation layer on top of standardized perinatal depression screening guidance (PHQ-9 and EPDS).

Core Architecture Principle:
The approved suggestion library (data/questionnaire_suggestions.json and data/library_overrides.json) is the SOLE source of suggestions.
The model may ONLY select up to 4 suggestion IDs from approved candidate suggestions and order them to match user-selected theme tags (Mode A).
When rewording is enabled (Mode B), the model rewrites selected library sentences into the target language or simpler words.
The model NEVER invents advice, numbers, medicines, conditions, or clinical content.
If anything fails validation or raises an exception, the system falls back to the deterministic result.


2. CONFIGURATION FLAGS
----------------------
The system is controlled by three environment variables in .env:

A. GUIDANCE_AI_ENABLED (boolean: "true" | "false", default: "false")
   - Master switch for AI personalisation.
   - When false (default): The endpoint POST /api/questionnaire/guidance bypasses LLM calls completely and immediately returns deterministic library suggestions. The model is called 0 times.
   - When true: Mode A (selection and ordering) is activated for non-gated assessments.

B. GUIDANCE_AI_REWORD (boolean: "true" | "false", default: "false")
   - Switch for Mode B rephrasing/rewriting into Indian languages (Hindi, Bengali, Tamil, or simpler English).
   - OFF by default until a licensed perinatal clinician reviews and validates output quality for each language.
   - When false: Rewording node is bypassed. The endpoint returns approved library translations (data/questionnaire_suggestions.json) in the requested language.
   - When true: Model generates rewrites for selected items only. Each rewrite must pass 7 automated safety checks (length, digits, URLs/phones, forbidden diagnostic words, medication words, script validation).

C. GUIDANCE_CACHE_DAYS (integer, default: 7)
   - Cache retention window in days for guidance responses.
   - Cached responses are keyed by:
     hash(instrument + band_id + stage + language + sorted_themes + mode + library_version_hash)
   - Any manual edit or addition to data/questionnaire_suggestions.json or data/library_overrides.json immediately alters library_version_hash, invalidating the entire cache automatically.


3. CLINICIAN REVIEW PROCESS FOR AI OUTPUT
-----------------------------------------
Before enabling GUIDANCE_AI_REWORD="true" in production or approving drafted suggestions:

A. Safety & Escalation Verification:
   1. Confirm that any assessment triggering safety (PHQ-9 Q9 > 0 or EPDS Q10 > 0) never calls the AI model and immediately shows emergency and Tele-MANAS (14416) helplines.
   2. Verify that high-distress bands ("moderate", "moderately_severe", "severe") always place the clinician contact recommendation (SUG_CLINICIAN_CONTACT) in the first position.
   3. Verify that relaxation practices marked as "mixed" evidence (SUG_CALMING_PRACTICE) never appear in moderate or severe bands.

B. Linguistic & Cultural Validation:
   1. Audit translated or reworded sentences against config/forbidden_words.json and config/medication_words.json.
   2. Ensure tone is supportive, warm, non-judgmental, and non-diagnostic. The wording must never imply a psychiatric diagnosis (e.g. "You are depressed") or promise specific outcomes.
   3. Check script compliance: Hindi text must use Devanagari script, Bengali text must use Bengali script, Tamil text must use Tamil script.

C. Monitoring Telemetry via Admin Route:
   Clinicians and administrators can inspect anonymized system telemetry (counts only, no user content) via:
   GET /api/admin/guidance/stats (Header: x-admin-token: <ADMIN_TOKEN>)
   Returns:
     - ai_selection_used: Count of successful Mode A selections served.
     - fallback: Count of deterministic fallbacks triggered.
     - validation_failure: Breakdown of validation rejections by reason (e.g. reword_digit, reword_phone, reword_forbidden_word, reword_medication).
     - cache_hit: Count of requests served from cache.


4. OFFLINE LIBRARY DRAFTING TOOL
--------------------------------
New library suggestions must never be authored directly by untracked prompt runs.
An offline CLI tool is provided for drafting: scripts/draft_library_item.py.

Rules & Safeguards:
- Offline and admin-only: Never runs in the user-facing request flow.
- Non-destructive: NEVER modifies data/questionnaire_suggestions.json.
- Output location: Writes draft entries ONLY to data/drafts/<id>.json.
- Draft status: The status field is ALWAYS set to "draft".
- Clinical sign-off notice: Prints "A clinician must review and approve this before use."
- Mandatory source: Refuses execution if source text/abstract is not provided.
- Safe networking: Makes zero web search or scraping calls (only the structured model API).

How to Draft a New Library Suggestion:
1. Save the source guideline excerpt, RCT abstract, or clinical trial summary in a .txt file (e.g. /tmp/who_perinatal_guideline.txt).
2. Run the script:
   ./.venv/bin/python scripts/draft_library_item.py --source-file /tmp/who_perinatal_guideline.txt --source-id "WHO-2023-REC"
3. The script asks the model to structure the recommendation into the library schema (id, category, English text <= 30 words, evidence level, evidence note quoting source, source_refs).
4. The draft file is generated at data/drafts/<generated_id>.json.
5. A clinician reviews the draft:
   - Verifies evidence quote matches source.
   - Inspects English text (<30 words, safe, non-diagnostic).
   - Approves translations.
6. Once clinically approved, an administrator moves the item to data/questionnaire_suggestions.json or data/library_overrides.json with status="approved".
