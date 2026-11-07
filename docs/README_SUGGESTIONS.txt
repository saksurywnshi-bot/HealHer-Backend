HEALHER THEME-SPECIFIC SENTIMENT SUGGESTIONS SYSTEM
====================================================

1. OVERVIEW
-----------
The HealHer mood reflection (sentiment) feature provides mothers with supportive,
evidence-based, theme-specific, tone-aware, and stage-aware self-care suggestions.

Key Principles:
- The AI model (Gemini) NEVER writes, generates, or edits suggestions.
  The model's sole responsibility is classifying note tone (6 fixed tones),
  intensity, themes (from 18 fixed clinical themes), risk language, and writing
  a validating reflection.
- Suggestions are chosen entirely by plain, deterministic code in
  app/services/suggestion_selector.py from the curated library file
  data/sentiment_suggestions.json (70 items based on clinical literature).
- The original library file is immutable. Clinician approvals and wording edits
  are saved into data/sentiment_overrides.json.
- No note text, themes, reflections, or scores are EVER stored in the suggestion
  history database. Only suggestion IDs and timestamps are kept to power recency.


2. SELECTION RULES & PIPELINE
-----------------------------
When a mother submits a check-in note to POST /api/sentiment/analyze, the selector
executes the following 10 clinical rules:

Rule 1 (Safety Gating):
  If tone is 'distressed', risk_language is true, or escalation is anything
  other than 'none':
  - The selector is NOT called.
  - suggestions is strictly [] (empty list).
  - The response returns the safety support block with verified national helplines:
    Tele-MANAS (14416), Emergency (112), Women Helpline (181).
    Disallowed foreign numbers (1800-889-4357) and legacy numbers (102, 108) never appear.

Rule 2 (Candidate Pool Filtering):
  Served candidates must meet:
  - Stage match: item.stage == user_stage OR item.stage == 'both'.
  - Tone match: user_tone in item.tones.
  - Theme match: item.generic is true OR at least one item.themes in detected themes.

Rule 3 (Primary Theme):
  The primary theme is the first detected theme returned by the model that has at
  least one qualifying non-generic candidate in the pool.
  If no themes were detected, generic items only are considered.

Rule 4 (Scoring Weights):
  Each qualifying candidate is scored:
  - Primary theme match: +3 points
  - Secondary detected theme match: +1 point
  - Primary tone match (tone is the first entry in item.tones): +2 points
  - Stage-specific item (item.stage == user_stage, not 'both'): +1 point
  - Not shown to this user in the last SUGGESTION_RECENCY_DAYS days: +2 points
  - Effort fit:
    Where tone is 'tired_overwhelmed' and effort is in ('none', '1min', '5min', 'talk'): +1 point
    Where tone is 'tired_overwhelmed' and effort is 'longer': -2 points

Rule 5 (Deterministic Tie-Breaking Shuffle):
  Ties are broken using a deterministic pseudo-random shuffle seeded by
  SHA-256(user_ref + local_date). This guarantees that recommendations for a user
  are completely stable across multiple requests on the same day, while varying
  naturally across different days.

Rule 6 (Result Composition & Shape):
  The selector builds a result list of up to 3 suggestions:
  - First slot: The highest-scoring item of kind 'understand' (normalising / validation)
    if one exists in the pool.
  - Subsequent slots: The highest-scoring items of kind 'do' (action) or 'talk' (support).
  - Clinician Limit: At most ONE item of kind 'clinician' per result.
  - Generic Limit: At most ONE generic item, and ONLY if fewer than 3 theme-specific
    items qualified.
  - No duplicate suggestion IDs are ever returned in a single response.

Rule 7 (Recency Filtering):
  If SUGGESTION_RECENCY_ENABLED=true, IDs shown to this user within the last
  SUGGESTION_RECENCY_DAYS days (default: 14) are excluded when sufficient non-recent
  candidates exist. If excluding them would cause the pool to drop below 3 items,
  the least recently shown items are re-admitted in order of oldest impression.

Rule 8 (Localization):
  The selector picks text_<language> (hi, bn, ta) if present and non-empty;
  otherwise, it falls back to text_en and sets text_language to 'en'.

Rule 9 (Explanation Metadata - why_this):
  Each theme-specific suggestion includes:
  - why_this_theme: English label from theme_labels_en (e.g. "Because you wrote about your sleep").
  - why_this_key: Translation key for frontend UI (e.g. "suggestion.why.sleep").
  Generic items omit why_this metadata (values are null / omitted).

Rule 10 (Safe Empty Fallback):
  If the candidate pool is empty (e.g. no approved items exist for that combination),
  the API returns suggestions: [] alongside the validating reflection and standard
  tone guidance. It never invents advice or falls back to legacy files.


3. RECENCY & SUGGESTION HISTORY
-------------------------------
To avoid repetitive suggestions, impressions are recorded in SQLite:
  Table: suggestion_history
  Columns:
    - id (INTEGER PRIMARY KEY)
    - user_ref (TEXT NOT NULL, indexed)
    - suggestion_id (TEXT NOT NULL, indexed)
    - shown_at (TEXT NOT NULL, indexed)

Environment Variables:
  SUGGESTION_RECENCY_DAYS=14
  SUGGESTION_RECENCY_ENABLED=true

Privacy Guarantees:
  - suggestion_history stores ONLY the hashed user identifier (user_ref),
    the suggestion identifier (suggestion_id), and an ISO-8601 timestamp.
  - Note texts, detected themes, sentiment reflections, and scores are NEVER stored
    or logged in this table or anywhere else in the backend.
  - Users can delete their suggestion history at any time via:
    DELETE /api/me/suggestion-history (or DELETE /api/sentiment/me/suggestion-history).


4. CLINICIAN APPROVAL & WORKFLOW
--------------------------------
The library data/sentiment_suggestions.json ships with 70 evidence-grounded suggestions.
15 items have needs_source = true and are in 'draft' status.

Clinician Rules:
- Only items with status 'approved' are served in production.
- In development, setting DEV_INCLUDE_DRAFTS=true and APP_ENV=development allows
  testing draft items.
- An item CANNOT be transitioned to 'approved' if needs_source is true or source_refs
  is empty. Attempting to approve an unsourced item returns HTTP 400.
- All suggestion texts must be strictly non-stigmatising and non-medical:
  - No digits allowed.
  - No diagnostic terms allowed (depression, depressed, mild, moderate, severe, diagnosis, disorder).
  - No medication terms allowed (tablet, medicine, antidepressant, SSRI, etc.).

Admin API Workflow (Requires x-admin-token):
1. Review Library & Coverage:
   GET /api/admin/sentiment-library
   Returns all items, evidence levels, sources, and coverage per theme and stage.
2. Check Untranslated Items:
   GET /api/admin/sentiment-library/untranslated
   Returns items missing Hindi, Bengali, or Tamil translations.
3. Update Status or Translations:
   PATCH /api/admin/sentiment-library/suggestion/{id}
   Body: {"status": "approved", "text_hi": "...", "text_bn": "...", "text_ta": "..."}
   Edits are saved into data/sentiment_overrides.json and validated immediately.
4. Privacy-Preserving Aggregate Stats:
   GET /api/admin/sentiment-library/stats
   Returns 30-day suggestion impressions, 14-day repeat rate, and empty result count.
   Zero user identifiers or note data are included.


5. MIGRATION & LEGACY CLEANUP
-----------------------------
The old suggestion files have been decommissioned and moved out of active paths:
- app/config/theme_to_suggestions.json -> legacy/theme_to_suggestions.json
- data/suggestions.json -> legacy/suggestions.json

The frontend API contract preserves backward compatibility:
- The new array field is `suggestions: [{ id, kind, text, text_language, why_this_theme, why_this_key }]`.
- The legacy single string field `suggestion` is populated with the text of the
  first suggestion for backward compatibility during client migration.
- `library_version` exposes the 16-character SHA-256 hash of active library content.
