================================================================================
HEAL-HER EDUCATIONAL VIDEO LIBRARY GUIDE
================================================================================

1. OVERVIEW
-----------
The Heal-Her Educational Video Library provides non-diagnostic, localized audio-visual
content on maternal recovery, postpartum wellbeing, bonding, and infant nutrition.
Videos are educational; they never trigger scoring, diagnostic bands, or safety actions.
No analytics, user tracking, or video watch history are ever logged.


2. DIRECTORY LAYOUT & FOLDER CONVENTION
---------------------------------------
Under the configured VIDEO_DIR (e.g. data/videos/):

  data/videos/
    <video_id>/
      en.mp4       (English video, primary fallback)
      hi.mp4       (Hindi video)
      bn.mp4       (Bengali video)
      ta.mp4       (Tamil video)
      en.vtt       (English subtitles, WEBVTT format, optional)
      hi.vtt       (Hindi subtitles, WEBVTT format, optional)
      bn.vtt       (Bengali subtitles, WEBVTT format, optional)
      ta.vtt       (Tamil subtitles, WEBVTT format, optional)
      poster.jpg   (Thumbnail/poster image, optional)

Rules:
- Video IDs must use lowercase alphanumeric characters and hyphens only (e.g. general-info-ppd).
- Video files must be .mp4 container with H.264 video and AAC audio.
- Subtitle files must be .vtt and start with the header 'WEBVTT'.
- Language codes must be one of: en, hi, bn, ta.
- File names can be direct (e.g. hi.mp4, hi.vtt) or prefixed (e.g. PPD_hindi.mp4, PPD_hi.vtt).


3. HOW TO ADD A NEW VIDEO OR LANGUAGE
-------------------------------------
Step 1: Prepare the video file
  Ensure the video is exported as an MP4 (H.264 video, AAC audio).
  Use the compression helper script if the file is large (see section 4).

Step 2: Place the file in the video folder
  Create data/videos/<video_id>/ if it does not already exist.
  Name the video file <language>.mp4 (e.g. ta.mp4) or <video_id>_<language>.mp4.
  If subtitle files are available, name them <language>.vtt (e.g. ta.vtt).

Step 3: Add or update the catalogue entry
  Edit data/video_catalogue.json:
  Add an entry under "videos" with:
    - "id": "<video_id>"
    - "audiences": ["postnatal", "caregiver"] (or "prenatal")
    - "stage": "postnatal" (or "prenatal")
    - "trimester": null (or 1, 2, 3 for prenatal)
    - "order": integer sort order
    - "topic": one of body, hormones, mood, baby, bonding, recovery, feeding, sleep, support, general
    - "titles": {"en": "...", "hi": "...", "bn": "...", "ta": "..."}
    - "descriptions": {"en": "...", "hi": "...", "bn": "...", "ta": "..."}
    - "allow_condition_words": false (set to true ONLY for condition-overview videos like PPD)

Step 4: Refresh the backend library
  Send a POST request to /api/admin/videos/refresh with header:
    x-admin-token: <your_admin_token>


4. VIDEO COMPRESSION SCRIPT (scripts/compress_video.sh)
-------------------------------------------------------
A manual helper script is provided at scripts/compress_video.sh.
It compresses raw video to web-optimized 720p H.264 with AAC audio and faststart metadata
(enabling immediate streaming without downloading the whole file first).

Command:
  ./scripts/compress_video.sh input_raw.mp4 output_web.mp4

Internal ffmpeg command:
  ffmpeg -i "$1" -vf scale=-2:720 -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 96k -movflags +faststart "$2"

Note: This helper is run by hand by an operator. The web backend never runs ffmpeg or transcodes files.


5. CATALOGUE FORMAT & CLINICAL VOCABULARY RULES
-----------------------------------------------
File location: data/video_catalogue.json

Schema:
{
  "version": "0.1",
  "videos": [
    {
      "id": "mother-child-nutrition",
      "audiences": ["postnatal", "caregiver"],
      "stage": "postnatal",
      "trimester": null,
      "order": 4,
      "topic": "feeding",
      "titles": {
        "en": "Mother and child nutrition",
        "hi": "",
        "bn": "",
        "ta": ""
      },
      "descriptions": {
        "en": "",
        "hi": "",
        "bn": "",
        "ta": ""
      },
      "allow_condition_words": false
    }
  ]
}

Clinical Word Rules:
- Titles and descriptions are written by humans only, never by AI.
- They must NOT contain forbidden non-diagnostic terms (e.g. mild, moderate, severe,
  depressed, depression, diagnosis, disorder, mental illness) or medication words (e.g. tablet,
  pill, medicine, sertraline, etc.).
- Controlled exception: Condition-focused videos (such as PPD overview) can set
  "allow_condition_words": true. This exempts ONLY the words "PPD", "postpartum", and
  "depression" in that entry's title and description. Severity words and medication words
  remain strictly refused.
- Any entry violating these rules is refused at startup and reported to the clinician.


6. ADMIN REPORT & CLINICAL OVERSIGHT
------------------------------------
GET /api/admin/videos/report
Header: x-admin-token: <ADMIN_TOKEN>

The report provides:
1. Video inventory: Languages and subtitles present and missing for each video.
2. File warnings: Any media file exceeding VIDEO_MAX_MB_WARN (default 50 MB).
3. Codec inspection: Warns if ffprobe detects non-standard codecs.
4. Refused entries: Catalogue entries blocked due to forbidden words.
5. Clinician review list: Entries utilizing the allow_condition_words flag.
6. Uncatalogued folders: Folders found on disk with no corresponding catalogue entry.
7. Missing folders: Catalogue entries with no media files on disk.
8. Fallback inventory: Videos that currently fall back to English for non-English users.

