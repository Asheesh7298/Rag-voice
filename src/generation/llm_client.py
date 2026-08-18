"""
LLM generation with an enforced structured output schema. On a malformed/unparseable
response we retry once with a corrective message before the harness falls back to a
safe decline -- this is the "retries + structured I/O" part of the harness requirement.

Supports two providers, selected via LLM_PROVIDER in .env:
  - "anthropic" -> Claude Messages API
  - "gemini"    -> Google Generative Language API (generateContent)
"""
from __future__ import annotations
import json
import time
import httpx
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.harness.schemas import GeneratedAnswer

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
You will be given a user query and a set of numbered context passages retrieved from a
knowledge base. Answer ONLY using information present in the passages. If the passages
do not contain enough information to answer, say so explicitly in the answer field and
set confidence low.

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{"answer": "<string>", "used_source_indices": [<ints of passages you actually used>], "confidence": <float 0-1>}
"""


class GenerationError(Exception):
    pass


class LLMClient:
    def __init__(self, timeout_s: float = 15.0):
        self.timeout_s = timeout_s
        self.provider = settings.llm_provider.lower()

    def _build_context_block(self, sources: list[str]) -> str:
        return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(sources))

    # ---------- Anthropic ----------

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
    def _call_anthropic(self, query: str, context_block: str, retry_note: str = "") -> str:
        headers = {
            "x-api-key": settings.llm_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": settings.llm_model,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_block}{retry_note}"}
            ],
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []))

    # ---------- Gemini ----------
    # REST shape: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY
    # System instructions go in a separate `system_instruction` field (not a message role).
    # Response: candidates[0].content.parts[*].text

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
    def _call_gemini(self, query: str, context_block: str, retry_note: str = "") -> str:
        model = settings.llm_model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": settings.llm_api_key}
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Query: {query}\n\nContext:\n{context_block}{retry_note}"}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.2,
                # Ask Gemini to enforce JSON output natively -- reduces retry rate.
                "responseMimeType": "application/json",
            },
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, params=params, json=body)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                # Common cause: response blocked by safety filters (finishReason == "SAFETY")
                reason = data.get("promptFeedback", {}).get("blockReason", "no candidates returned")
                raise GenerationError(f"Gemini returned no candidates: {reason}")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

    # ---------- shared ----------

    def _call(self, query: str, context_block: str, retry_note: str = "") -> str:
        if self.provider == "gemini":
            return self._call_gemini(query, context_block, retry_note)
        return self._call_anthropic(query, context_block, retry_note)

    def _parse(self, raw_text: str) -> GeneratedAnswer:
        cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return GeneratedAnswer(**parsed)

    def generate(self, query: str, sources: list[str]) -> tuple[GeneratedAnswer, float]:
        context_block = self._build_context_block(sources)
        t0 = time.perf_counter()
        try:
            raw = self._call(query, context_block)
            answer = self._parse(raw)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            # One corrective retry covers both malformed JSON and valid JSON
            # with an invalid schema (for example, confidence outside 0..1).
            try:
                raw = self._call(
                    query, context_block,
                    retry_note="\n\n(Your previous response was not valid JSON matching the schema. Reply with ONLY the JSON object.)"
                )
                answer = self._parse(raw)
            except Exception as retry_error:
                raise GenerationError(
                    f"LLM returned an invalid structured response after retry: {retry_error}"
                ) from retry_error
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e
        t1 = time.perf_counter()
        return answer, round((t1 - t0) * 1000, 2)
