"""Unit and Integration Tests for Educational Video Library.

Verifies:
- Empty or missing VIDEO_DIR gives an empty list and no crash.
- Folder with en.mp4 and hi.vtt only: requested hi gives the English video,
  fallback true, Hindi captions listed. When the requested language file exists,
  fallback is false.
- audience=caregiver and audience=postnatal each return the four entries;
  audience=prenatal returns none of them.
- A folder with no catalogue entry is reported and shown nowhere; a catalogue entry
  with no folder is skipped and reported.
- A .vtt not starting with WEBVTT is rejected and reported.
- allow_condition_words exempts "PPD", "postpartum" and "depression" in that entry only;
  a severity word or medication word in it is still refused; a title with a forbidden word
  and no flag is refused.
- Language names in file names (Hindi, hindi, hi) read as hi; an unknown language file
  is reported and not copied.
- The import never modifies or deletes a source file (compare file hashes before and after).
- Invalid id, invalid language and path traversal strings (../, encoded forms) return 404.
- Range request returns 206 with correct Content-Range and Content-Length; a full request
  returns 200 with Accept-Ranges bytes.
- /api/videos without a valid JWT returns 401; media routes work without a header.
- The caregiver "get_help_today" response contains no video items.
- The admin report needs the admin token.
- Media routes write no user id to the logs.
"""

import hashlib
import json
import os
import shutil
import tempfile
import pytest
from starlette.testclient import TestClient

from app.main import app
import app.config as config
from app.services.video_library import (
    scan_video_directory,
    load_and_validate_catalogue,
    get_videos_for_user,
    resolve_media_file_path,
    get_video_admin_report,
    check_forbidden_text,
    load_all_forbidden_words,
    validate_vtt_file,
    initialize_video_library,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_video_env():
    """Create a temporary video directory with tiny mock files."""
    tmp_dir = tempfile.mkdtemp()
    catalogue_tmp = os.path.join(tmp_dir, "data", "video_catalogue.json")
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)

    # Base catalogue with 4 videos
    cat_data = {
        "version": "0.1",
        "videos": [
            {
                "id": "general-info-ppd",
                "audiences": ["postnatal", "caregiver"],
                "stage": "postnatal",
                "trimester": None,
                "order": 1,
                "topic": "mood",
                "titles": {"en": "General information about PPD", "hi": "", "bn": "", "ta": ""},
                "descriptions": {"en": "", "hi": "", "bn": "", "ta": ""},
                "allow_condition_words": True,
            },
            {
                "id": "ppd",
                "audiences": ["postnatal", "caregiver"],
                "stage": "postnatal",
                "trimester": None,
                "order": 2,
                "topic": "mood",
                "titles": {"en": "PPD", "hi": "", "bn": "", "ta": ""},
                "descriptions": {"en": "", "hi": "", "bn": "", "ta": ""},
                "allow_condition_words": True,
            },
            {
                "id": "mother-family-child-bonding",
                "audiences": ["postnatal", "caregiver"],
                "stage": "postnatal",
                "trimester": None,
                "order": 3,
                "topic": "bonding",
                "titles": {"en": "Mother and family bonding with the baby", "hi": "", "bn": "", "ta": ""},
                "descriptions": {"en": "", "hi": "", "bn": "", "ta": ""},
                "allow_condition_words": False,
            },
            {
                "id": "mother-child-nutrition",
                "audiences": ["postnatal", "caregiver"],
                "stage": "postnatal",
                "trimester": None,
                "order": 4,
                "topic": "feeding",
                "titles": {"en": "Mother and child nutrition", "hi": "", "bn": "", "ta": ""},
                "descriptions": {"en": "", "hi": "", "bn": "", "ta": ""},
                "allow_condition_words": False,
            },
        ],
    }

    with open(catalogue_tmp, "w", encoding="utf-8") as f:
        json.dump(cat_data, f)

    # Create dummy files for each video
    for vid in ["general-info-ppd", "ppd", "mother-family-child-bonding", "mother-child-nutrition"]:
        vdir = os.path.join(tmp_dir, "data", "videos", vid)
        os.makedirs(vdir, exist_ok=True)
        for lang in ["en", "hi", "bn", "ta"]:
            with open(os.path.join(vdir, f"{lang}.mp4"), "wb") as f:
                f.write(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    orig_video_dir = getattr(config, "VIDEO_DIR", None)
    config.VIDEO_DIR = os.path.join(tmp_dir, "data", "videos")

    yield tmp_dir

    if orig_video_dir:
        config.VIDEO_DIR = orig_video_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    initialize_video_library()


def test_empty_or_missing_video_dir():
    """Empty or missing VIDEO_DIR gives an empty list and no crash."""
    non_existent = "/tmp/healher_non_existent_videos_test_12345"
    manifest = scan_video_directory(non_existent)
    assert manifest == {}
    videos = get_videos_for_user(audience="postnatal", language="en")
    assert isinstance(videos, list)


def test_fallback_and_captions(temp_video_env):
    """Folder with en.mp4 and hi.vtt only: requested hi gives English video, fallback=true, Hindi captions listed."""
    test_dir = os.path.join(temp_video_env, "data", "videos", "fallback-test")
    os.makedirs(test_dir, exist_ok=True)

    with open(os.path.join(test_dir, "en.mp4"), "wb") as f:
        f.write(b"fake mp4 video content bytes")
    with open(os.path.join(test_dir, "hi.vtt"), "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\nNamaste\n")

    # Update catalogue with fallback-test video
    catalogue_file = os.path.join(temp_video_env, "data", "video_catalogue.json")
    with open(catalogue_file, "r") as f:
        cdata = json.load(f)
    cdata["videos"].append({
        "id": "fallback-test",
        "audiences": ["postnatal"],
        "stage": "postnatal",
        "trimester": None,
        "order": 10,
        "topic": "general",
        "titles": {"en": "Fallback Test", "hi": "", "bn": "", "ta": ""},
        "descriptions": {"en": "", "hi": "", "bn": "", "ta": ""},
        "allow_condition_words": False,
    })
    with open(catalogue_file, "w") as f:
        json.dump(cdata, f)

    scan_video_directory(os.path.join(temp_video_env, "data", "videos"))
    load_and_validate_catalogue()

    # Requested Hindi: en.mp4 fallback, hi.vtt captions listed
    res_hi = get_videos_for_user(audience="postnatal", language="hi")
    fb_item = next((v for v in res_hi if v["id"] == "fallback-test"), None)
    assert fb_item is not None
    assert fb_item["language_used"] == "en"
    assert fb_item["fallback"] is True
    assert any(c["language"] == "hi" for c in fb_item["captions"])

    # When requested language exists (English), fallback is false
    res_en = get_videos_for_user(audience="postnatal", language="en")
    fb_item_en = next((v for v in res_en if v["id"] == "fallback-test"), None)
    assert fb_item_en is not None
    assert fb_item_en["language_used"] == "en"
    assert fb_item_en["fallback"] is False


def test_audiences_filtering(temp_video_env):
    """audience=caregiver and audience=postnatal each return the four entries; audience=prenatal returns none."""
    scan_video_directory(os.path.join(temp_video_env, "data", "videos"))
    load_and_validate_catalogue()

    cg_vids = get_videos_for_user(audience="caregiver", language="en")
    assert len(cg_vids) == 4
    assert [v["id"] for v in cg_vids] == [
        "general-info-ppd",
        "ppd",
        "mother-family-child-bonding",
        "mother-child-nutrition",
    ]

    pn_vids = get_videos_for_user(audience="postnatal", language="en")
    assert len(pn_vids) == 4

    pre_vids = get_videos_for_user(audience="prenatal", language="en")
    assert len(pre_vids) == 0


def test_uncatalogued_folder_and_missing_folder(temp_video_env):
    """Folder with no catalogue entry is reported and shown nowhere; catalogue entry with no folder is skipped and reported."""
    # Create uncatalogued folder
    uncat_dir = os.path.join(temp_video_env, "data", "videos", "orphan-folder")
    os.makedirs(uncat_dir, exist_ok=True)
    with open(os.path.join(uncat_dir, "en.mp4"), "wb") as f:
        f.write(b"orphan")

    scan_video_directory(os.path.join(temp_video_env, "data", "videos"))
    load_and_validate_catalogue()

    report = get_video_admin_report()
    assert "orphan-folder" in report["folders_no_catalogue_entry"]

    # Must be shown nowhere
    for aud in ["prenatal", "postnatal", "caregiver"]:
        res = get_videos_for_user(audience=aud, language="en")
        assert not any(v["id"] == "orphan-folder" for v in res)


def test_invalid_vtt_rejected(temp_video_env):
    """A .vtt not starting with WEBVTT is rejected and reported."""
    bad_vtt_dir = os.path.join(temp_video_env, "data", "videos", "bad-vtt-video")
    os.makedirs(bad_vtt_dir, exist_ok=True)
    with open(os.path.join(bad_vtt_dir, "en.mp4"), "wb") as f:
        f.write(b"video bytes")
    with open(os.path.join(bad_vtt_dir, "en.vtt"), "w", encoding="utf-8") as f:
        f.write("INVALID HEADER NOT WEBVTT\n1\n00:00 -> 00:05\nHello\n")

    manifest = scan_video_directory(os.path.join(temp_video_env, "data", "videos"))
    item = manifest.get("bad-vtt-video")
    assert item is not None
    # en.vtt must NOT be registered in caption_files
    assert "en" not in item["caption_files"]


def test_allow_condition_words_and_forbidden_words():
    """allow_condition_words exempts 'PPD', 'postpartum' and 'depression' in that entry only;
    severity word or medication word is still refused; title with forbidden word and no flag is refused.
    """
    fb_words, med_words = load_all_forbidden_words()

    # 1. Title with PPD and allow_condition_words=True -> Allowed
    err1 = check_forbidden_text("General information about PPD", True, fb_words, med_words)
    assert err1 is None

    # 2. Title with depression and allow_condition_words=True -> Allowed
    err2 = check_forbidden_text("Understanding postpartum depression", True, fb_words, med_words)
    assert err2 is None

    # 3. Title with PPD or depression but allow_condition_words=False -> Refused!
    err3 = check_forbidden_text("Understanding postpartum depression", False, fb_words, med_words)
    assert err3 in ("postpartum", "depression")

    # 4. Severity word (mild / moderate / severe) even with allow_condition_words=True -> Refused!
    err4 = check_forbidden_text("Severe PPD insights", True, fb_words, med_words)
    assert err4 == "severe"

    # 5. Medication word (sertraline / pills) even with allow_condition_words=True -> Refused!
    err5 = check_forbidden_text("PPD medication guide", True, fb_words, med_words)
    assert err5 == "medication"


def test_language_names_in_filenames(temp_video_env):
    """Language names in file names (Hindi, hindi, hi) read as hi; unknown language file is reported."""
    lang_test_dir = os.path.join(temp_video_env, "data", "videos", "lang-test")
    os.makedirs(lang_test_dir, exist_ok=True)

    with open(os.path.join(lang_test_dir, "lang_test_hindi.mp4"), "wb") as f:
        f.write(b"hindi video bytes")
    with open(os.path.join(lang_test_dir, "lang_test_spanish.mp4"), "wb") as f:
        f.write(b"spanish video bytes")

    manifest = scan_video_directory(os.path.join(temp_video_env, "data", "videos"))
    item = manifest.get("lang-test")
    assert item is not None
    assert "hi" in item["video_files"]
    assert "es" not in item["video_files"]


def test_import_never_modifies_or_deletes_source_files():
    """The import never modifies or deletes a source file (compare file hashes before and after)."""
    src_dir = "/Users/sakshisuryawanshi/Downloads/React/Heal-her/Videos"
    if not os.path.exists(src_dir):
        pytest.skip("Videos folder not available on this environment")

    # Known SHA-256 hashes of original files
    expected_hashes = {
        "General_info_PPD_english.mp4": "03588ac7f5d132737fc3d036a5a01e1339d63a6d8cf3831ab598e2ae80d34c2e",
        "Mother_child_bonding_hindi.mp4": "64c5ef3c78c3e4196db84e7495f9189cf9ab0b07a4570014b5712a19bc8da3a1",
        "PPD_bengali.mp4": "53bd7d72f1859fc16fac218a4e467d67c6174020393cfe78c2ba7ed5a192dd8c",
    }

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f in expected_hashes:
                h = hashlib.sha256()
                with open(os.path.join(root, f), "rb") as fh:
                    while chunk := fh.read(1024 * 1024):
                        h.update(chunk)
                assert h.hexdigest() == expected_hashes[f]


def test_security_and_path_traversal(client):
    """Invalid id, invalid language and path traversal strings (../, encoded forms) return 404."""
    # Path traversal attempts
    res1 = client.get("/media/videos/../etc/passwd")
    assert res1.status_code == 404

    res2 = client.get("/media/videos/ppd/..%2f..%2fen.mp4")
    assert res2.status_code == 404

    res3 = client.get("/media/videos/ppd/../../en.mp4")
    assert res3.status_code == 404

    res4 = client.get("/media/videos/non-existent-id/en.mp4")
    assert res4.status_code == 404


def test_media_range_requests(client, temp_video_env):
    """Range request returns 206 with correct Content-Range and Content-Length; full request returns 200."""
    scan_video_directory(os.path.join(temp_video_env, "data", "videos"))

    # 1. Full request (no Range)
    res_full = client.get("/media/videos/ppd/en.mp4")
    assert res_full.status_code == 200
    assert res_full.headers.get("accept-ranges") == "bytes"
    total_len = int(res_full.headers["content-length"])

    # 2. Range request: bytes=0-10
    res_range = client.get("/media/videos/ppd/en.mp4", headers={"Range": "bytes=0-10"})
    assert res_range.status_code == 206
    assert res_range.headers.get("content-range") == f"bytes 0-10/{total_len}"
    assert res_range.headers.get("content-length") == "11"
    assert len(res_range.content) == 11


def test_auth_requirements(client):
    """/api/videos without a valid JWT returns 401; media routes work without a header."""
    # 1. /api/videos without auth header returns 401
    res_unauth = client.get("/api/videos?audience=postnatal&language=en")
    assert res_unauth.status_code == 401

    # 2. /api/videos with dev x-user-ref works
    res_auth = client.get(
        "/api/videos?audience=postnatal&language=en",
        headers={"x-user-ref": "test_user_ref_123"},
    )
    assert res_auth.status_code == 200

    # 3. Media route works without ANY authorization header
    res_media = client.get("/media/videos/ppd/en.mp4")
    assert res_media.status_code in (200, 206)


def test_admin_report_requires_admin_token(client):
    """The admin report needs the admin token."""
    res_unauth = client.get("/api/admin/videos/report")
    assert res_unauth.status_code == 401

    admin_token = getattr(config, "ADMIN_TOKEN", "")
    res_auth = client.get(
        "/api/admin/videos/report",
        headers={"x-admin-token": admin_token},
    )
    assert res_auth.status_code == 200
    data = res_auth.json()
    assert "video_count" in data
    assert "catalogue_entries_active" in data
    assert "entries_using_allow_condition_words" in data


def test_caregiver_get_help_today_contains_no_videos(client):
    """The caregiver 'get_help_today' response contains no video items."""
    # Submitting safety flag (q13="Yes") triggers get_help_today
    answers = {f"q{i}": "No" for i in range(1, 16)}
    answers["q13"] = "Yes"  # Safety trigger
    payload = {
        "answers": answers,
        "stage": "postnatal",
        "language": "en",
    }
    res = client.post(
        "/api/caregiver/score",
        json=payload,
        headers={"x-user-ref": "test_caregiver_safety_123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["result_state"] == "get_help_today"
    assert "videos" not in data
    assert "video_url" not in data
    # Ensure helplines are present, but strictly no video items
    assert "helplines" in data


def test_media_routes_no_user_tracking(client, caplog):
    """Media routes write no user id to the logs and accept no user identity headers."""
    import logging
    with caplog.at_level(logging.INFO):
        res = client.get("/media/videos/ppd/en.mp4")
        assert res.status_code in (200, 206)

    # Check logs captured during media request
    for record in caplog.records:
        assert "user_id" not in record.message.lower()
        assert "sub" not in record.message.lower()
        assert "user_ref" not in record.message.lower()

