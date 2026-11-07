"""AI4Bharat IndicConformer STT Adapter for Indic languages (Hindi, Bengali, Tamil).

Privacy & Safety:
- Audio content and transcripts are NEVER logged.
- Only device selection and system lifecycle events are logged.
"""

import logging
import threading
from typing import Any, Optional

from app.config import (
    INDIC_CONFORMER_MODEL_ID,
    INDIC_CONFORMER_DECODING,
    HF_TOKEN,
    STT_DEVICE,
)
from app.services.stt.base import BaseSTTAdapter

logger = logging.getLogger("healher_backend.stt.indic_conformer")


class IndicConformerAdapter(BaseSTTAdapter):
    """Adapter for Hindi, Bengali, and Tamil speech-to-text using AI4Bharat IndicConformer."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        decoding_mode: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self._model_id = model_id or INDIC_CONFORMER_MODEL_ID
        self._decoding_mode = (decoding_mode or INDIC_CONFORMER_DECODING).lower()
        self._hf_token = hf_token or HF_TOKEN
        self._model = None
        self._device: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return "indic-conformer"

    def _determine_device(self) -> Any:
        import torch

        pref = STT_DEVICE.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "cpu":
            return torch.device("cpu")

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model(self):
        """Lazily load and cache the IndicConformer model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                device = self._determine_device()
                self._device = device
                logger.info(
                    "Loading IndicConformer model '%s' on device: %s (decoding: %s)",
                    self._model_id,
                    device.type,
                    self._decoding_mode,
                )
                from transformers import AutoModel

                try:
                    kwargs = {"trust_remote_code": True}
                    if self._hf_token:
                        kwargs["token"] = self._hf_token

                    model = AutoModel.from_pretrained(self._model_id, **kwargs)
                    model = model.to(device)
                    model.eval()
                    self._model = model
                    logger.info("IndicConformer model '%s' loaded successfully.", self._model_id)
                except Exception as e:
                    err_msg = str(e)
                    if "gated" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                        logger.error(
                            "Failed to access gated repository '%s'. Please accept terms on Hugging Face "
                            "and set HF_TOKEN in your .env file.",
                            self._model_id,
                        )
                    raise

        return self._model

    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe 16 kHz mono WAV audio file to Indic text.

        Args:
            audio_path: Path to the 16 kHz mono WAV audio file.
            language: Language code ('hi', 'bn', or 'ta').

        Returns:
            Transcribed text.
        """
        import torch

        model = self._get_model()
        device = self._device or self._determine_device()

        # Load audio using torchaudio or soundfile
        try:
            import torchaudio

            wav, sr = torchaudio.load(audio_path)
            # Ensure mono channel
            if wav.ndim > 1 and wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            # Ensure 16000 Hz sample rate
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
        except Exception:
            import soundfile as sf

            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data).unsqueeze(0)

        wav = wav.to(device)

        with torch.no_grad():
            output = model(wav, language, self._decoding_mode)

        if isinstance(output, (list, tuple)):
            text = " ".join(str(s) for s in output).strip()
        else:
            text = str(output).strip()

        return text
