"""
Sarvam speech-to-text client -- verified against live API docs
(https://docs.sarvam.ai/api-reference/speech-to-text/transcribe) on 2026-08-13.

Endpoint: POST /speech-to-text, multipart form, header `api-subscription-key`.
Response: {request_id, transcript, language_code, timestamps?, language_probability?}

Important: language_code must be BCP-47 (e.g. "hi-IN"), not our internal 2-letter
dataset codes ("hi") -- LANG_TO_BCP47 below maps between them. Pass language_code=None
(or omit) to let the API auto-detect.
"""
from __future__ import annotations
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

# Our internal dataset uses bare ISO codes (matches settings.languages); Sarvam
# wants BCP-47. Extend this if you add languages beyond the current 13.
LANG_TO_BCP47 = {
    "as": "as-IN", "bn": "bn-IN", "gu": "gu-IN", "hi": "hi-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN",
    "or": "od-IN",  # note: Sarvam uses "od-IN" for Odia, not "or-IN"
    "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN", "ur": "ur-IN",
}


class STTError(Exception):
    pass


class SarvamSTTClient:
    def __init__(self, timeout_s: float = 8.0, model: str = "saaras:v3"):
        self.timeout_s = timeout_s
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
    def _call(self, audio_bytes: bytes, language_code: str | None) -> dict:
        headers = {"api-subscription-key": settings.sarvam_api_key}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": self.model, "mode": "transcribe"}
        if language_code:
            data["language_code"] = language_code
        # else: omit entirely -> Sarvam auto-detects language

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(settings.sarvam_stt_url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            return resp.json()

    def transcribe(self, audio_bytes: bytes, language_code: str | None = None) -> dict:
        """
        language_code: our internal 2-letter code (e.g. "hi") or None for auto-detect.
        Returns {"transcript": str, "language_detected": str | None, "latency_ms": float}
        """
        bcp47 = LANG_TO_BCP47.get(language_code) if language_code else None
        t0 = time.perf_counter()
        try:
            result = self._call(audio_bytes, bcp47)
        except Exception as e:
            raise STTError(f"STT failed after retries: {e}") from e
        t1 = time.perf_counter()

        return {
            "transcript": result.get("transcript", ""),
            "language_detected": result.get("language_code", bcp47),
            "latency_ms": round((t1 - t0) * 1000, 2),
        }