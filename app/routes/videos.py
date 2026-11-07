"""Video Library Routes for HealHer Backend.

Endpoints:
- GET  /api/videos (JWT required, rate limited)
- GET  /media/videos/{video_id}/{filename} (Public, HTTP Range 206 supported, Cache-Control 3600)
- POST /api/admin/videos/refresh (Admin token required)
- GET  /api/admin/videos/report (Admin token required)
"""

import os
import re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.config import RATE_LIMIT, ADMIN_TOKEN
from app.limiter import limiter
from app.services.auth import get_current_user_ref
from app.services.video_library import (
    get_videos_for_user,
    resolve_media_file_path,
    initialize_video_library,
    get_video_admin_report,
)

router = APIRouter(tags=["videos"])


def verify_admin_token(x_admin_token: Optional[str] = Header(None, alias="x-admin-token")) -> None:
    """Verify that x-admin-token header matches configured ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_TOKEN is not configured on the server.",
        )
    if not x_admin_token or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
        )


@router.get("/api/videos", summary="Retrieve educational videos for requested audience & language")
@limiter.limit(RATE_LIMIT)
def list_videos(
    request: Request,
    audience: str = Query(..., pattern="^(prenatal|postnatal|caregiver)$"),
    language: str = Query("en", pattern="^(en|hi|bn|ta)$"),
    trimester: Optional[int] = Query(None, ge=1, le=3),
    user_ref: str = Depends(get_current_user_ref),
):
    """Retrieve educational videos for the specified audience and language.
    
    Requires Supabase JWT identity. Applies language fallback to English with
    captions when localized video is unavailable.
    """
    videos = get_videos_for_user(audience=audience, language=language, trimester=trimester)
    return {"count": len(videos), "videos": videos}


def stream_file_with_range(file_path: str, content_type: str, request: Request) -> Response:
    """Stream a local media file supporting HTTP Range (206 Partial Content)."""
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Media file not found.")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range") or request.headers.get("range")

    headers = {
        "Content-Type": content_type,
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }

    if not range_header:
        headers["Content-Length"] = str(file_size)

        def iter_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk

        return StreamingResponse(iter_full(), status_code=200, headers=headers)

    # Parse Range: bytes=start-end, bytes=start-, bytes=-suffix
    range_match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
    if not range_match:
        headers["Content-Range"] = f"bytes */{file_size}"
        return Response(status_code=416, headers=headers)

    start_str, end_str = range_match.groups()
    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
    elif start_str:
        start = int(start_str)
        end = file_size - 1
    elif end_str:
        start = max(0, file_size - int(end_str))
        end = file_size - 1
    else:
        headers["Content-Range"] = f"bytes */{file_size}"
        return Response(status_code=416, headers=headers)

    if start >= file_size or start > end:
        headers["Content-Range"] = f"bytes */{file_size}"
        return Response(status_code=416, headers=headers)

    end = min(end, file_size - 1)
    content_length = end - start + 1

    headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    headers["Content-Length"] = str(content_length)

    def iter_range():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(remaining, 64 * 1024)
                data = f.read(chunk_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(iter_range(), status_code=206, headers=headers)


@router.get("/media/videos/{video_id}/{filename:path}")
@router.head("/media/videos/{video_id}/{filename:path}")
def serve_media_file(
    video_id: str,
    filename: str,
    request: Request,
):
    """Serve public, read-only educational video files and captions.
    
    Supports HTTP Range (206) for seeking and mobile HTML5 playback.
    No cookies, no analytics, no user ids logged.
    """
    # Strict validation against path traversal
    if ".." in video_id or "/" in video_id or "\\" in video_id:
        raise HTTPException(status_code=404, detail="Invalid video ID")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="Invalid media file name")

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".mp4":
        content_type = "video/mp4"
    elif ext == ".vtt":
        content_type = "text/vtt"
    elif ext in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif ext == ".png":
        content_type = "image/png"
    else:
        raise HTTPException(status_code=404, detail="Unsupported media format")

    file_path = resolve_media_file_path(video_id, filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="Media file not found in library manifest")

    return stream_file_with_range(file_path, content_type, request)


@router.post("/api/admin/videos/refresh", summary="Rescan VIDEO_DIR and revalidate catalogue")
def refresh_video_library(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Rescan VIDEO_DIR and refresh in-memory manifest."""
    verify_admin_token(x_admin_token)
    initialize_video_library()
    report = get_video_admin_report()
    return {
        "status": "refreshed",
        "video_count": report["video_count"],
        "catalogue_entries_active": report["catalogue_entries_active"],
    }


@router.get("/api/admin/videos/report", summary="Clinical and administrative video report")
def video_admin_report(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Retrieve full clinical and administrative report of video library."""
    verify_admin_token(x_admin_token)
    return get_video_admin_report()

