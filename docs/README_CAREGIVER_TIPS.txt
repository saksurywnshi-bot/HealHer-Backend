HEALHER CAREGIVER 'HOW YOU CAN HELP HER TODAY' DAILY TIPS (V0.2)
================================================================

1. OVERVIEW
-----------
The Caregiver Track provides five stage-appropriate, practical tips every day to partners and family members supporting women during their perinatal journey.
Version 0.2 expands the library to 86 curated tips, introducing three new interaction kinds:
- check_in: Gentle conversational questions to understand how she is feeling.
- small_joy: Thoughtful gestures and small comforts to lift her day.
- try_together: Low-pressure activities to do together with no expectations.

Zero AI is used for tip generation or rewording. Tips are chosen strictly via deterministic plain Python code from data/caregiver_suggestions.json and clinician overrides in data/caregiver_overrides.json.


2. SELECTION RULES & DAILY MIX
------------------------------
Each day, the selection engine (app/services/caregiver_daily.py) selects exactly CAREGIVER_DAILY_COUNT (5) tips:
1. Stage Isolation:
   - If the woman is prenatal, only tips tagged "prenatal" or "both" are eligible.
   - If the woman is postnatal, only tips tagged "postnatal" or "both" are eligible.
   - Prenatal users never see postnatal-only tips, and postnatal users never see prenatal-only tips.

2. Safety Override (Result State: get_help_today):
   - If safety questions are triggered (Q13, Q14, or Q15 in observation log), result_state becomes get_help_today.
   - In get_help_today, NO tips are returned (help_today: []).
   - Only verified Indian helplines are returned:
     * Tele-MANAS: 14416 (toll-free, 24/7)
     * Women Helpline: 181
     * National Emergency: 112
     * Forbidden helpline numbers (1800-889-4357, 102, 108) are strictly banned.

3. Favoured Themes:
   - For result_state talk_soon: at least one tip from themes "appointments" or "start_conversation" is prioritised.
   - For result_state extra_support: flagged areas from the questionnaire map to favoured themes via config/caregiver_flags.json:
     * money_food -> share_chores, ask_others
     * tension -> calm_home
     * isolated -> help_her_connect
     * baby_hard -> share_baby_care
     * no_support -> ask_others, share_chores, help_her_connect
     * signs_high -> appointments, start_conversation, notice_changes

4. Daily Mix Composition:
   The 5 tips follow this structured daily mix:
   - Tip 1: One favoured-theme tip (if requested by state or flags)
   - Tip 2: Exactly one of kind "check_in"
   - Tip 3: Exactly one of light kinds ("small_joy" or "try_together")
   - Tip 4: Exactly one of kind "do"
   - Tip 5: Exactly one of kind "say"
   - Wildcard slot: Seeded coin (even SHA-256 hash of user_ref | stage | day | 'wild') selects an "aware" kind (notice, avoid, self_care, or clinician) if none is yet present, otherwise fills from remaining eligible tips.

5. Strict Caps per Day:
   - at most 1 each of kinds: check_in, self_care, clinician, notice, avoid
   - at most 2 of kind: say
   - at most 3 of kind: do
   - at most 2 of kinds: small_joy and try_together combined
   - Theme variety: No two tips with the same first theme on the same day.
   - No duplicate tip IDs on the same day.

6. Daily Stability & Midnight Shuffle:
   - Selection and presentation ordering are deterministically seeded using SHA-256(user_ref | stage | local_date | tip_id).
   - The user sees the exact same 5 tips throughout the calendar day (Asia/Kolkata).
   - After midnight in Asia/Kolkata, a new fresh set of 5 tips is calculated.


3. VARIETY WINDOW & RECENCY
---------------------------
- CAREGIVER_RECENCY_DAYS (default 7 days):
  * Tips not shown in the last 7 days are ranked first.
  * Tips shown within the last 7 days are pushed to the back, ranked by oldest impression first.
- Over a 90-day simulation:
  * No tip returns sooner than 6 days after it was shown.
  * Over 90% of served tips in the library appear at least once.
  * Check-in questions return after approximately 7-8 days, providing periodic gentle prompts.


4. HOW A CLINICIAN APPROVES & EDITS TIPS
----------------------------------------
All tips in data/caregiver_suggestions.json begin in "draft" status.
During development (APP_ENV=development and DEV_INCLUDE_DRAFTS=true), drafts are served to enable testing.
In production, only tips with status "approved" are served to users.

Clinician Workflow:
1. Inspect Library & Coverage:
   GET /api/admin/caregiver-library (Header: x-admin-token: <ADMIN_TOKEN>)
   Returns all tips, their evidence levels, sources, and a coverage analysis for prenatal and postnatal stages.
   Warns if any stage has fewer than 35 served tips, fewer than 7 check_in, fewer than 14 small_joy+try_together, or fewer than 7 say tips.

2. Approve or Edit a Tip:
   PATCH /api/admin/caregiver-library/tip/{id}
   Headers: x-admin-token: <ADMIN_TOKEN>
   Body:
   {
     "status": "approved",
     "text_hi": "...",
     "text_bn": "...",
     "text_ta": "...",
     "text_en": "..."
   }
   Rules & Safeguards:
   - Approvals are REFUSED if needs_source is true or source_refs is empty.
   - Any edits to text_en must NOT contain digits (0-9) or forbidden clinical/medication words.
   - All clinician edits are merged into data/caregiver_overrides.json. The base library data/caregiver_suggestions.json is NEVER modified.
   - Re-validation runs automatically on patch, updating the library version hash.

3. Find Untranslated Tips:
   GET /api/admin/caregiver-library/untranslated
   Returns all tips that lack Hindi (hi), Bengali (bn), or Tamil (ta) translations.

4. View Aggregate Usage:
   GET /api/admin/caregiver-library/stats
   Returns frequency counts of tips shown over the last 30 days and count of days with fewer than 5 tips.


5. PRIVACY & DATA PROTECTION
----------------------------
- The caregiver_tip_history table in translations.db stores ONLY:
  * user_ref (64-character SHA-256 hash of user sub)
  * tip_id (e.g. CG_P_APPT)
  * shown_on (date string YYYY-MM-DD)
  * created_at (ISO timestamp)
- Questionnaire answers, themes, risk counts, and clinical scores are NEVER stored in the database or written to server logs.
- User Data Deletion:
  * Users can delete all their caregiver tip history via DELETE /api/me/caregiver-tip-history.
  * Calling the root deletion route DELETE /api/me/suggestion-history also purges caregiver tip history automatically.

