"""Educational Video Library Service & Manifest Manager for HealHer.

Features:
- Scans VIDEO_DIR at startup and on manual refresh (POST /api/admin/videos/refresh).
- In-memory manifest of validated MP4 videos and WEBVTT captions.
- Zero crash on empty or missing video folders.
- Strict clinical vocabulary validation (forbidden non-diagnostic terms and medication words).
- Controlled exception: allow_condition_words=True exempts 'PPD', 'postpartum', and 'depression'
  only for condition-overview entries.
- Streaming HTTP Range (206 Partial Content) support for mobile and desktop video seeking.
- Privacy guarantee: No personal watch logs, no user identifiers in media routes.
"""

import copy
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

import app.config as config

logger = logging.getLogger("healher_backend.video_library")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOGUE_PATH = os.path.join(_BASE_DIR, "data", "video_catalogue.json")
FORBIDDEN_WORDS_PATH = os.path.join(_BASE_DIR, "app", "config", "forbidden_words.json")
MEDICATION_WORDS_PATH = os.path.join(_BASE_DIR, "app", "config", "medication_words.json")

SUPPORTED_LANGUAGES = {"en", "hi", "bn", "ta"}
LANGUAGE_NAMES_MAP = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "bn": "bn",
    "bengali": "bn",
    "bangla": "bn",
    "ta": "ta",
    "tamil": "ta",
}
LANGUAGE_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
}

# In-memory storage of scanned manifest and catalogue
_MANIFEST: Dict[str, Dict[str, Any]] = {}
_CATALOGUE: List[Dict[str, Any]] = []
_REFUSED_CATALOGUE_ENTRIES: List[Dict[str, Any]] = []
_ENTRIES_USING_CONDITION_WORDS: List[str] = []
_CODEC_WARNINGS: Dict[str, List[str]] = {}
_UNKNOWN_FILES: Dict[str, List[str]] = {}
_FILES_OVER_WARN: Dict[str, List[Dict[str, Any]]] = {}


def load_all_forbidden_words() -> Tuple[Set[str], Set[str]]:
    """Load base forbidden words and medication words across all languages.
    
    Returns:
        (forbidden_non_diagnostic_words, medication_words)
    """
    forbidden: Set[str] = {
        "mild", "moderate", "severe", "depressed", "depression",
        "diagnosis", "diagnos", "diagnose", "disorder", "clinical",
        "bipolar", "psychosis", "schizo", "syndrome", "pathology", "mental illness"
    }
    if os.path.exists(FORBIDDEN_WORDS_PATH):
        try:
            with open(FORBIDDEN_WORDS_PATH, "r", encoding="utf-8") as f:
                fb = json.load(f)
                for lang_words in fb.values():
                    if isinstance(lang_words, list):
                        for w in lang_words:
                            forbidden.add(w.strip().lower())
        except Exception as e:
            logger.warning("Could not read forbidden_words.json: %s", e)

    medications: Set[str] = set()
    if os.path.exists(MEDICATION_WORDS_PATH):
        try:
            with open(MEDICATION_WORDS_PATH, "r", encoding="utf-8") as f:
                mw = json.load(f)
                for lang_words in mw.values():
                    if isinstance(lang_words, list):
                        for w in lang_words:
                            medications.add(w.strip().lower())
        except Exception as e:
            logger.warning("Could not read medication_words.json: %s", e)

    return forbidden, medications


def check_forbidden_text(
    text: str,
    allow_condition_words: bool,
    forbidden_words: Set[str],
    medication_words: Set[str],
) -> Optional[str]:
    """Check if a text string contains forbidden or medication words.
    
    If allow_condition_words=True, exempts only 'ppd', 'postpartum', and 'depression'.
    Severity words ('mild', 'moderate', 'severe'), diagnosis words, and medication words
    are NEVER exempted.

    Returns:
        The violating forbidden/medication word found, or None if clean.
    """
    if not text:
        return None

    lower_text = text.lower()
    # Normalize punctuation into spaces for exact word boundary matching
    words = re.findall(r"[\w]+", lower_text)
    word_set = set(words)

    # 1. Check medication words (strictly forbidden in all contexts)
    for med in medication_words:
        if " " in med:
            if med in lower_text:
                return med
        elif med in word_set:
            return med

    # 2. Check forbidden non-diagnostic terms
    exempt_condition_words = {
        "ppd", "postpartum", "depression",
        "प्रसवोत्तर", "अवसाद", "डिप्रेशन",
        "বিষণ্ণতা", "বিষণ্নতা", "ডিপ্রেশন",
        "மனச்சோர்வு", "பிரசவத்திற்குப் பிந்தைய"
    }
    for fb in forbidden_words:
        if allow_condition_words and fb in exempt_condition_words:
            continue
        if " " in fb:
            if fb in lower_text:
                return fb
        elif fb in word_set:
            return fb

    return None


def resolve_video_search_dir(configured_dir: Optional[str] = None) -> str:
    """Resolve the directory to search for videos.
    
    Checks:
    1. custom configured_dir if provided
    2. VIDEO_DIR setting in app.config
    3. VIDEO_DIR/data/videos if present
    4. Backend_heal_her/data/videos fallback
    """
    if configured_dir is not None:
        sub_data_videos = os.path.join(configured_dir, "data", "videos")
        if os.path.isdir(sub_data_videos):
            return sub_data_videos
        return configured_dir

    target = getattr(config, "VIDEO_DIR", "")
    if target:
        sub_data_videos = os.path.join(target, "data", "videos")
        if os.path.isdir(sub_data_videos):
            return sub_data_videos
        if os.path.isdir(target):
            return target

    # Fallback to backend data/videos
    backend_data_videos = os.path.join(_BASE_DIR, "data", "videos")
    return backend_data_videos


def validate_vtt_file(file_path: str) -> bool:
    """Validate that a .vtt subtitle file starts with 'WEBVTT'."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(64).strip()
            # Handle UTF-8 BOM if present
            header = header.lstrip("\ufeff").strip()
            return header.startswith("WEBVTT")
    except Exception as e:
        logger.warning("Error reading subtitle file %s: %s", file_path, e)
        return False


def check_codecs_with_ffprobe(file_path: str) -> Tuple[bool, str]:
    """Check if an MP4 video has H.264 video and AAC audio using ffprobe if available.
    
    Returns:
        (is_valid_or_uncheckable, description_message)
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return True, "ffprobe not installed on system; MP4 container validated"

    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "stream=codec_name,codec_type",
            "-of", "json",
            file_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if res.returncode != 0:
            return False, f"ffprobe error reading {os.path.basename(file_path)}: {res.stderr.strip()}"

        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        video_codecs = [s.get("codec_name") for s in streams if s.get("codec_type") == "video"]
        audio_codecs = [s.get("codec_name") for s in streams if s.get("codec_type") == "audio"]

        is_h264 = any(c == "h264" for c in video_codecs)
        is_aac = any(c == "aac" for c in audio_codecs) or len(audio_codecs) == 0

        if not is_h264 or not is_aac:
            return False, f"Non-standard codecs in {os.path.basename(file_path)}: video={video_codecs}, audio={audio_codecs}"

        return True, "H.264 video with AAC audio"
    except Exception as e:
        return False, f"ffprobe check exception: {str(e)}"


def scan_video_directory(search_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Scan the video directory and build an in-memory manifest.
    
    Validates:
    - .mp4 files for videos
    - .vtt files for subtitles (must start with WEBVTT)
    - poster.jpg if present
    - File sizes against VIDEO_MAX_MB_WARN
    - Never crashes on missing or empty folders
    """
    root_path = resolve_video_search_dir(search_dir)
    manifest: Dict[str, Dict[str, Any]] = {}
    codec_warnings: Dict[str, List[str]] = {}
    unknown_files: Dict[str, List[str]] = {}
    files_over_warn: Dict[str, List[Dict[str, Any]]] = {}

    warn_bytes = getattr(config, "VIDEO_MAX_MB_WARN", 50) * 1024 * 1024

    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        logger.info("Video directory '%s' does not exist or is empty.", root_path)
        return manifest

    try:
        entries = sorted(os.listdir(root_path))
    except Exception as e:
        logger.warning("Unable to list video directory %s: %s", root_path, e)
        return manifest

    for folder_name in entries:
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        raw_id = folder_name.strip().lower().replace("_", "-").replace(" ", "-")
        # Standardize known aliases
        if raw_id == "mother-child-bonding":
            video_id = "mother-family-child-bonding"
        elif raw_id in ("trimester1", "pre-trimester-1", "pre_trimester_1"):
            video_id = "trimester-1"
        elif raw_id in ("trimester2", "pre-trimester-2", "pre_trimester_2"):
            video_id = "trimester-2"
        elif raw_id in ("trimester3", "pre-trimester-3", "pre_trimester_3"):
            video_id = "trimester-3"
        else:
            video_id = raw_id

        # Validate video id syntax (lowercase letters, digits, hyphens)
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", video_id):
            logger.warning("Skipping invalid video folder name: %s", folder_name)
            continue

        video_files: Dict[str, str] = {}
        caption_files: Dict[str, str] = {}
        poster_file: Optional[str] = None
        file_sizes: Dict[str, int] = {}
        folder_unknown: List[str] = []
        folder_over_warn: List[Dict[str, Any]] = []
        folder_codec_warn: List[str] = []

        try:
            folder_contents = sorted(os.listdir(folder_path))
        except Exception:
            folder_contents = []

        for fname in folder_contents:
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue

            sz = os.path.getsize(fpath)
            file_sizes[fname] = sz

            if sz > warn_bytes:
                folder_over_warn.append({
                    "file_name": fname,
                    "size_mb": round(sz / (1024 * 1024), 2),
                    "warn_limit_mb": getattr(config, "VIDEO_MAX_MB_WARN", 50),
                })

            base_name, ext = os.path.splitext(fname)
            ext_lower = ext.lower()

            # Identify language from filename
            # Expected names: en.mp4, hi.mp4, General_info_PPD_english.mp4, etc.
            lang_code: Optional[str] = None
            for key, code in LANGUAGE_NAMES_MAP.items():
                if base_name.lower() == key or base_name.lower().endswith(f"_{key}") or base_name.lower().endswith(f"-{key}"):
                    lang_code = code
                    break

            if ext_lower == ".mp4":
                if lang_code and lang_code in SUPPORTED_LANGUAGES:
                    video_files[lang_code] = fpath
                    # Codec inspection
                    ok, msg = check_codecs_with_ffprobe(fpath)
                    if not ok:
                        folder_codec_warn.append(msg)
                else:
                    folder_unknown.append(fname)
            elif ext_lower == ".vtt":
                if lang_code and lang_code in SUPPORTED_LANGUAGES:
                    if validate_vtt_file(fpath):
                        caption_files[lang_code] = fpath
                    else:
                        folder_unknown.append(f"{fname} (invalid WEBVTT header)")
                else:
                    folder_unknown.append(fname)
            elif ext_lower in (".jpg", ".jpeg", ".png") and "poster" in base_name.lower():
                poster_file = fpath
            else:
                if fname not in (".DS_Store", "Thumbs.db"):
                    folder_unknown.append(fname)

        manifest[video_id] = {
            "id": video_id,
            "folder_name": folder_name,
            "folder_path": folder_path,
            "video_files": video_files,
            "caption_files": caption_files,
            "poster_file": poster_file,
            "file_sizes": file_sizes,
        }

        if folder_unknown:
            unknown_files[video_id] = folder_unknown
        if folder_over_warn:
            files_over_warn[video_id] = folder_over_warn
        if folder_codec_warn:
            codec_warnings[video_id] = folder_codec_warn

    global _MANIFEST, _UNKNOWN_FILES, _FILES_OVER_WARN, _CODEC_WARNINGS
    _MANIFEST = manifest
    _UNKNOWN_FILES = unknown_files
    _FILES_OVER_WARN = files_over_warn
    _CODEC_WARNINGS = codec_warnings
    return manifest


def load_and_validate_catalogue(custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load video_catalogue.json and strictly validate against schema and forbidden words."""
    global _CATALOGUE, _REFUSED_CATALOGUE_ENTRIES, _ENTRIES_USING_CONDITION_WORDS
    _CATALOGUE = []
    _REFUSED_CATALOGUE_ENTRIES = []
    _ENTRIES_USING_CONDITION_WORDS = []

    catalogue_file = custom_path
    if not catalogue_file:
        vdir = getattr(config, "VIDEO_DIR", "")
        if vdir:
            cand1 = os.path.join(vdir, "data", "video_catalogue.json")
            cand2 = os.path.join(os.path.dirname(vdir), "video_catalogue.json")
            cand3 = os.path.join(vdir, "video_catalogue.json")
            if os.path.exists(cand1):
                catalogue_file = cand1
            elif os.path.exists(cand2):
                catalogue_file = cand2
            elif os.path.exists(cand3):
                catalogue_file = cand3

    if not catalogue_file or not os.path.exists(catalogue_file):
        catalogue_file = CATALOGUE_PATH

    if not os.path.exists(catalogue_file):
        logger.warning("video_catalogue.json not found at %s", catalogue_file)
        return []

    try:
        with open(catalogue_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.error("Failed to parse video_catalogue.json: %s", e)
        return []

    videos = raw.get("videos", [])
    if not isinstance(videos, list):
        logger.error("video_catalogue.json 'videos' key must be a list.")
        return []

    forbidden_words, medication_words = load_all_forbidden_words()

    valid_entries: List[Dict[str, Any]] = []

    for entry in videos:
        if not isinstance(entry, dict):
            continue

        vid_id = entry.get("id", "").strip().lower()
        if not vid_id:
            continue

        allow_condition = bool(entry.get("allow_condition_words", False))
        if allow_condition:
            _ENTRIES_USING_CONDITION_WORDS.append(vid_id)

        # Validate titles and descriptions against forbidden terms
        violating_word: Optional[str] = None
        titles = entry.get("titles", {})
        descriptions = entry.get("descriptions", {})

        for text in list(titles.values()) + list(descriptions.values()):
            if isinstance(text, str):
                vw = check_forbidden_text(text, allow_condition, forbidden_words, medication_words)
                if vw:
                    violating_word = vw
                    break

        if violating_word:
            _REFUSED_CATALOGUE_ENTRIES.append({
                "id": vid_id,
                "reason": f"Violates clinical forbidden word rule (contains: '{violating_word}')",
                "violating_word": violating_word,
                "entry": entry,
            })
            logger.warning(
                "Refusing video catalogue entry '%s': contains forbidden term '%s'",
                vid_id,
                violating_word,
            )
            continue

        valid_entries.append(entry)

    _CATALOGUE = valid_entries
    return valid_entries


def initialize_video_library() -> None:
    """Initialize or refresh video manifest and catalogue in memory."""
    scan_video_directory()
    load_and_validate_catalogue()
    logger.info(
        "Video library initialized with %d scanned folders and %d active catalogue entries.",
        len(_MANIFEST),
        len(_CATALOGUE),
    )


def get_videos_for_user(
    audience: str,
    language: str = "en",
    trimester: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query video library for an audience, applying language fallback and ordering.
    
    Language resolution rule:
    - If video file exists in requested language, use it (fallback=False).
    - Otherwise, use English video if it exists, adding requested language's captions
      if they exist (fallback=True).
    - If neither requested language nor English video exists, skip the video.
    
    Titles & descriptions use requested language if filled, else English.
    
    Ordering:
    - Ordered by trimester (if requested trimester matches, prioritized first), order, then id.
    """
    lang = language.lower() if language else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    items: List[Dict[str, Any]] = []

    for cat in _CATALOGUE:
        audiences = cat.get("audiences", [])
        if audience not in audiences:
            continue

        vid_id = cat.get("id")
        manifest_item = _MANIFEST.get(vid_id)
        if not manifest_item:
            # Check alias
            if vid_id == "mother-family-child-bonding" and "mother-child-bonding" in _MANIFEST:
                manifest_item = _MANIFEST["mother-child-bonding"]
            elif vid_id in ("trimester-1", "trimester1", "pre_trimester_1"):
                manifest_item = _MANIFEST.get("trimester-1") or _MANIFEST.get("trimester1")
            elif vid_id in ("trimester-2", "trimester2", "pre_trimester_2"):
                manifest_item = _MANIFEST.get("trimester-2") or _MANIFEST.get("trimester2")
            elif vid_id in ("trimester-3", "trimester3", "pre_trimester_3"):
                manifest_item = _MANIFEST.get("trimester-3") or _MANIFEST.get("trimester3")

        if not manifest_item:
            # Catalogue entry with no folder on disk is skipped
            continue

        video_files = manifest_item.get("video_files", {})
        caption_files = manifest_item.get("caption_files", {})

        # Determine video source
        language_used: Optional[str] = None
        fallback: bool = False

        if lang in video_files:
            language_used = lang
            fallback = False
        elif "en" in video_files:
            language_used = "en"
            fallback = (lang != "en")
        else:
            # Neither requested language nor English exists; skip item
            continue

        video_url = f"/media/videos/{vid_id}/{language_used}.mp4"

        # Captions
        captions: List[Dict[str, str]] = []
        for c_lang in sorted(caption_files.keys()):
            captions.append({
                "language": c_lang,
                "label": LANGUAGE_LABELS.get(c_lang, c_lang.title()),
                "url": f"/media/videos/{vid_id}/{c_lang}.vtt",
            })

        # Poster
        poster_url = f"/media/videos/{vid_id}/poster.jpg" if manifest_item.get("poster_file") else None

        # Titles and descriptions
        titles = cat.get("titles", {})
        descriptions = cat.get("descriptions", {})

        title_text = titles.get(lang) or titles.get("en") or ""
        desc_text = descriptions.get(lang) or descriptions.get("en") or ""

        # Title review flag: if title was generated or empty
        title_needs_review = not bool(titles.get(lang)) if lang != "en" else False

        item_entry = {
            "id": vid_id,
            "title": title_text,
            "description": desc_text,
            "stage": cat.get("stage", "postnatal"),
            "trimester": cat.get("trimester"),
            "topic": cat.get("topic", "general"),
            "audiences": audiences,
            "order": cat.get("order", 99),
            "title_needs_review": title_needs_review,
            "language_used": language_used,
            "fallback": fallback,
            "video_url": video_url,
            "poster_url": poster_url,
            "captions": captions,
            "available_languages": sorted(video_files.keys()),
        }
        items.append(item_entry)

    # Sort items:
    # If trimester is specified, put videos matching that trimester first
    def sort_key(item: Dict[str, Any]) -> Tuple[int, int, int, str]:
        tri_val = item.get("trimester")
        tri_priority = 0 if (trimester is not None and tri_val == trimester) else 1
        tri_order = tri_val if tri_val is not None else 999
        order_val = item.get("order", 99)
        return (tri_priority, tri_order, order_val, item.get("id", ""))

    items.sort(key=sort_key)
    return items


def resolve_media_file_path(video_id: str, filename: str) -> Optional[str]:
    """Securely resolve media file path strictly from in-memory manifest.
    
    Prevents path traversal, path normalization tricks, and uncatalogued access.
    """
    manifest_item = _MANIFEST.get(video_id)
    if not manifest_item:
        if video_id == "mother-family-child-bonding" and "mother-child-bonding" in _MANIFEST:
            manifest_item = _MANIFEST["mother-child-bonding"]
        elif video_id in ("trimester-1", "trimester1", "pre_trimester_1"):
            manifest_item = _MANIFEST.get("trimester-1") or _MANIFEST.get("trimester1")
        elif video_id in ("trimester-2", "trimester2", "pre_trimester_2"):
            manifest_item = _MANIFEST.get("trimester-2") or _MANIFEST.get("trimester2")
        elif video_id in ("trimester-3", "trimester3", "pre_trimester_3"):
            manifest_item = _MANIFEST.get("trimester-3") or _MANIFEST.get("trimester3")

    if not manifest_item:
        return None

    folder_path = manifest_item["folder_path"]
    candidate = os.path.abspath(os.path.join(folder_path, filename))

    # Strict path traversal check
    if not candidate.startswith(os.path.abspath(folder_path)):
        return None

    if os.path.isfile(candidate):
        return candidate

    return None


def get_video_admin_report() -> Dict[str, Any]:
    """Generate clinical and system administration report for video library."""
    cat_by_id = {c["id"]: c for c in _CATALOGUE}
    all_video_ids = sorted(set(list(_MANIFEST.keys()) + list(cat_by_id.keys())))

    per_video: Dict[str, Any] = {}
    fallback_videos: List[Dict[str, Any]] = []

    for vid_id in all_video_ids:
        manifest_item = _MANIFEST.get(vid_id)
        cat_item = cat_by_id.get(vid_id)

        video_files = manifest_item.get("video_files", {}) if manifest_item else {}
        caption_files = manifest_item.get("caption_files", {}) if manifest_item else {}

        languages_present = sorted(video_files.keys())
        languages_missing = sorted(SUPPORTED_LANGUAGES - set(languages_present))

        captions_present = sorted(caption_files.keys())
        captions_missing = sorted(SUPPORTED_LANGUAGES - set(captions_present))

        audiences = cat_item.get("audiences", []) if cat_item else []

        over_warn = _FILES_OVER_WARN.get(vid_id, [])
        codec_warn = _CODEC_WARNINGS.get(vid_id, [])
        unknown = _UNKNOWN_FILES.get(vid_id, [])

        per_video[vid_id] = {
            "in_catalogue": cat_item is not None,
            "on_disk": manifest_item is not None,
            "audiences": audiences,
            "languages_present": languages_present,
            "languages_missing": languages_missing,
            "captions_present": captions_present,
            "captions_missing": captions_missing,
            "files_over_warn_limit": over_warn,
            "codec_warnings": codec_warn,
            "unknown_files": unknown,
        }

        # Check fallback status for each non-English language
        if "en" in languages_present:
            for l in ["hi", "bn", "ta"]:
                if l not in languages_present:
                    fallback_videos.append({
                        "id": vid_id,
                        "language": l,
                        "fallback_to": "en",
                        "has_captions": l in captions_present,
                    })

    folders_no_catalogue = [v for v in _MANIFEST.keys() if v not in cat_by_id]
    catalogue_entries_no_folder = [
        c["id"] for c in _CATALOGUE if c["id"] not in _MANIFEST and (c["id"] != "mother-family-child-bonding" or "mother-child-bonding" not in _MANIFEST)
    ]

    return {
        "status": "healthy",
        "video_count": len(all_video_ids),
        "catalogue_entries_active": len(_CATALOGUE),
        "entries_refused_forbidden_words": _REFUSED_CATALOGUE_ENTRIES,
        "entries_using_allow_condition_words": _ENTRIES_USING_CONDITION_WORDS,
        "folders_no_catalogue_entry": folders_no_catalogue,
        "catalogue_entries_no_folder": catalogue_entries_no_folder,
        "fallback_videos": fallback_videos,
        "per_video_details": per_video,
    }
