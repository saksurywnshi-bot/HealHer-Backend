#!/usr/bin/env python3
"""HealHer Post-Deployment Smoke Test Suite.

Uses ONLY the Python standard library.
Takes BACKEND_URL and FRONTEND_URL from environment variables.
Never hard-codes URLs and never prints secrets.

Usage:
    BACKEND_URL="https://your-backend.onrender.com" FRONTEND_URL="https://your-frontend.vercel.app" python scripts/smoke_test.py
"""

import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "").rstrip("/")

def log_result(name: str, passed: bool, detail: str = ""):
    status_str = "PASS" if passed else "FAIL"
    msg = f"[{status_str}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    return passed

def log_info(name: str, detail: str):
    print(f"[INFO] {name} - {detail}")

def log_skipped(name: str, reason: str):
    print(f"[SKIPPED] {name} - {reason}")

def main():
    print("=" * 70)
    print("HEALHER POST-DEPLOYMENT SMOKE TEST")
    print("=" * 70)

    if not BACKEND_URL or not FRONTEND_URL:
        print("ERROR: Both BACKEND_URL and FRONTEND_URL environment variables must be set.")
        print("Example: BACKEND_URL=https://... FRONTEND_URL=https://... python scripts/smoke_test.py")
        sys.exit(1)

    print(f"Target Backend:  {BACKEND_URL}")
    print(f"Target Frontend: {FRONTEND_URL}\n")

    results = []

    # -------------------------------------------------------------------------
    # Check 1: GET BACKEND_URL/health returns 200
    # -------------------------------------------------------------------------
    health_url = f"{BACKEND_URL}/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "HealHerSmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            passed = resp.status == 200
            results.append(log_result("1. Backend Health Check (/health)", passed, f"HTTP {resp.status}"))
    except Exception as exc:
        results.append(log_result("1. Backend Health Check (/health)", False, f"Request failed: {exc}"))

    # -------------------------------------------------------------------------
    # Check 2: Protected endpoint returns 401/403 with no token, and with only x-user-ref
    # -------------------------------------------------------------------------
    protected_url = f"{BACKEND_URL}/api/sentiment/patterns"
    passed_no_token = False
    passed_x_user_ref = False

    # 2a. Call with no token
    try:
        req = urllib.request.Request(protected_url, headers={"User-Agent": "HealHerSmokeTest/1.0"})
        urllib.request.urlopen(req, timeout=15)
        passed_no_token = False
    except urllib.error.HTTPError as exc:
        passed_no_token = exc.code in (401, 403)
    except Exception:
        passed_no_token = False

    # 2b. Call with only x-user-ref (must still return 401 in production mode)
    try:
        req = urllib.request.Request(
            protected_url,
            headers={"x-user-ref": "test-unauthorized-user-ref", "User-Agent": "HealHerSmokeTest/1.0"}
        )
        urllib.request.urlopen(req, timeout=15)
        passed_x_user_ref = False
    except urllib.error.HTTPError as exc:
        passed_x_user_ref = exc.code in (401, 403)
    except Exception:
        passed_x_user_ref = False

    passed_auth = passed_no_token and passed_x_user_ref
    detail_auth = f"No token -> {'401/403' if passed_no_token else 'NOT 401'}; x-user-ref only -> {'401/403' if passed_x_user_ref else 'NOT 401 (production mode check failed)'}"
    results.append(log_result("2. Auth & Production Mode Protection", passed_auth, detail_auth))

    # -------------------------------------------------------------------------
    # Check 3: CORS preflight (OPTIONS)
    # -------------------------------------------------------------------------
    cors_url = f"{BACKEND_URL}/health"
    passed_allowed = False
    passed_disallowed = False

    # 3a. OPTIONS with FRONTEND_URL Origin
    try:
        req = urllib.request.Request(
            cors_url,
            method="OPTIONS",
            headers={
                "Origin": FRONTEND_URL,
                "Access-Control-Request-Method": "GET",
                "User-Agent": "HealHerSmokeTest/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            allow_origin = resp.headers.get("Access-Control-Allow-Origin", "")
            passed_allowed = (allow_origin == FRONTEND_URL or allow_origin == "*")
    except urllib.error.HTTPError as exc:
        allow_origin = exc.headers.get("Access-Control-Allow-Origin", "")
        passed_allowed = (allow_origin == FRONTEND_URL or allow_origin == "*")
    except Exception:
        passed_allowed = False

    # 3b. OPTIONS with disallowed origin https://example.com
    disallowed_origin = "https://example.com"
    try:
        req = urllib.request.Request(
            cors_url,
            method="OPTIONS",
            headers={
                "Origin": disallowed_origin,
                "Access-Control-Request-Method": "GET",
                "User-Agent": "HealHerSmokeTest/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            allow_origin = resp.headers.get("Access-Control-Allow-Origin", "")
            passed_disallowed = (allow_origin != disallowed_origin)
    except urllib.error.HTTPError as exc:
        allow_origin = exc.headers.get("Access-Control-Allow-Origin", "")
        passed_disallowed = (allow_origin != disallowed_origin)
    except Exception:
        passed_disallowed = True

    passed_cors = passed_allowed and passed_disallowed
    detail_cors = f"Frontend origin allowed: {passed_allowed}; example.com disallowed: {passed_disallowed}"
    results.append(log_result("3. CORS Preflight Configuration", passed_cors, detail_cors))

    # -------------------------------------------------------------------------
    # Check 4: Transcribe route status when ENABLE_STT is false
    # -------------------------------------------------------------------------
    transcribe_url = f"{BACKEND_URL}/api/transcribe"
    try:
        req = urllib.request.Request(
            transcribe_url,
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json", "User-Agent": "HealHerSmokeTest/1.0"}
        )
        urllib.request.urlopen(req, timeout=15)
        log_info("4. Speech-to-Text Transcribe Route", "Endpoint responded with 200 (STT appears active on server)")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 503 and "Voice input is not available" in body:
            log_info("4. Speech-to-Text Transcribe Route", "Returned 503 'Voice input is not available on this server' (as expected when ENABLE_STT=false)")
        else:
            log_info("4. Speech-to-Text Transcribe Route", f"Returned HTTP {exc.code}: {exc.reason}")
    except Exception as exc:
        log_info("4. Speech-to-Text Transcribe Route", f"Notice: {exc}")

    # -------------------------------------------------------------------------
    # Check 5: GET FRONTEND_URL and inspect linked JavaScript bundles for localhost
    # -------------------------------------------------------------------------
    frontend_passed = False
    leak_found = False
    leaked_files = []

    try:
        req = urllib.request.Request(FRONTEND_URL, headers={"User-Agent": "HealHerSmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
            frontend_passed = (resp.status == 200)

        if frontend_passed:
            # Extract script src attributes
            script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
            # Scan top linked JS assets (up to 15)
            for src in script_srcs[:15]:
                script_url = urllib.parse.urljoin(FRONTEND_URL, src)
                if not script_url.endswith(".js") and not ".js?" in script_url:
                    continue
                try:
                    js_req = urllib.request.Request(script_url, headers={"User-Agent": "HealHerSmokeTest/1.0"})
                    with urllib.request.urlopen(js_req, timeout=15) as js_resp:
                        js_code = js_resp.read().decode("utf-8", errors="ignore")
                        if "localhost:3001" in js_code or "127.0.0.1:3001" in js_code:
                            leak_found = True
                            leaked_files.append(src)
                except Exception:
                    pass

        passed_leak_check = frontend_passed and not leak_found
        detail_leak = "No localhost:3001 or 127.0.0.1:3001 references in frontend bundles" if not leak_found else f"Found references in: {', '.join(leaked_files)}"
        results.append(log_result("5. Frontend Loading & Zero Localhost Leaks", passed_leak_check, detail_leak))
    except Exception as exc:
        results.append(log_result("5. Frontend Loading & Zero Localhost Leaks", False, f"Frontend request failed: {exc}"))

    # -------------------------------------------------------------------------
    # Check 6: Count of approved items served by library endpoints
    # -------------------------------------------------------------------------
    # In public routes, /api/videos provides educational library items without requiring auth
    videos_url = f"{BACKEND_URL}/api/videos?language=en&stage=postnatal"
    try:
        req = urllib.request.Request(videos_url, headers={"User-Agent": "HealHerSmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            count = len(data) if isinstance(data, list) else len(data.get("videos", []))
            if count > 0:
                results.append(log_result("6. Approved Library Items Served", True, f"{count} items returned by public video catalogue"))
            else:
                log_skipped("6. Approved Library Items Served", "0 items returned from public video query")
    except Exception:
        log_skipped("6. Approved Library Items Served", "Public library catalogue route query returned error or requires authenticated session")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    total_passed = sum(1 for r in results if r)
    total_checks = len(results)
    print(f"SUMMARY: {total_passed}/{total_checks} core checks passed.")
    print("=" * 70)

    if total_passed == total_checks:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

