"""Audio validation and preprocessing utilities for Speech-to-Text.

Converts incoming audio (webm, wav, mp3, m4a, ogg) to 16 kHz mono WAV format
using ffmpeg or python-av / torchaudio / soundfile.

Privacy & Safety:
- Audio content is NEVER logged.
- Temporary files must be strictly cleaned up after processing.
"""

import os
import shutil
import subprocess
import logging
from typing import Optional, Set
from fastapi import HTTPException, status

logger = logging.getLogger("healher_backend.stt.audio_utils")

ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".webm", ".wav", ".mp3", ".m4a", ".ogg"}
MAX_AUDIO_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB limit
SUPPORTED_STT_LANGUAGES: Set[str] = {"hi", "bn", "ta", "en"}


def validate_audio_input(
    filename: Optional[str],
    content_type: Optional[str],
    size_bytes: int,
    language: str,
) -> None:
    """Validate incoming audio file type, size, and language code.

    Raises:
        HTTPException: If input is invalid, returning clear 4xx status codes.
    """
    # 1. Validate language code
    lang = (language or "").strip().lower()
    if lang not in SUPPORTED_STT_LANGUAGES:
        supported_str = ", ".join(sorted(SUPPORTED_STT_LANGUAGES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language '{language}'. Supported languages: {supported_str}",
        )

    # 2. Validate file size
    if size_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty.",
        )
    if size_bytes > MAX_AUDIO_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Audio file exceeds the maximum allowed size of {MAX_AUDIO_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # 3. Validate file extension or MIME type
    ext = os.path.splitext(filename or "")[1].lower()
    is_valid_ext = ext in ALLOWED_AUDIO_EXTENSIONS
    is_valid_mime = bool(
        content_type
        and (
            content_type.startswith("audio/")
            or content_type.startswith("video/webm")
            or content_type in ("application/octet-stream",)
        )
    )

    if not (is_valid_ext or is_valid_mime):
        allowed_str = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Supported formats: {allowed_str}",
        )


def convert_to_16k_mono_wav(input_path: str, output_wav_path: str) -> None:
    """Convert input audio to 16 kHz mono PCM WAV format.

    Uses system ffmpeg if available on PATH; otherwise falls back to
    PyAV (libav) and soundfile to ensure cross-platform compatibility.

    Args:
        input_path: Path to the input audio file.
        output_wav_path: Destination path for the converted WAV file.

    Raises:
        RuntimeError: If audio conversion fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_wav_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
                return
            logger.warning(
                "ffmpeg conversion returned non-zero code (%d), falling back to PyAV decoder.",
                result.returncode,
            )
        except Exception as e:
            logger.warning("ffmpeg execution failed (%s), falling back to PyAV decoder.", e)

    # Fallback using PyAV (libav)
    try:
        import av
        import numpy as np
        import soundfile as sf

        container = av.open(input_path)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            if resampled:
                for r in resampled:
                    frames.append(r.to_ndarray())

        if not frames:
            raise RuntimeError("No audio frames decoded from input file.")

        all_audio = np.concatenate(frames, axis=1)  # shape: (1, total_samples)
        sf.write(output_wav_path, all_audio.T, 16000, subtype="PCM_16")
    except Exception as av_err:
        raise RuntimeError(f"Failed to convert audio to 16 kHz mono WAV: {av_err}") from av_err
