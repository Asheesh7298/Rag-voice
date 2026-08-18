"""
The harness: a small explicit state machine, not a single prompt-in/text-out call.

States: RECEIVE -> [STT] -> SAFETY_GATE -> RETRIEVE -> CONFIDENCE_GATE -> GENERATE
        -> GROUNDEDNESS_GATE -> RESPOND

Each stage is timed independently and any failure/guardrail trip short-circuits to
a well-formed PipelineResponse rather than propagating an exception to the caller.
"""
from __future__ import annotations
import time
import logging

from src.config import settings
from src.harness.schemas import PipelineResponse, SourceChunk
from src.guardrails import checks
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, GenerationError
from src.stt.sarvam_client import SarvamSTTClient, STTError

logger = logging.getLogger("voice_rag.pipeline")


class Pipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, stt: SarvamSTTClient | None = None):
        self.retriever = retriever
        self.llm = llm
        self.stt = stt or SarvamSTTClient()

    def _decline(self, query: str, reason: str, timings: dict, transcript: str | None = None) -> PipelineResponse:
        logger.info("guardrail triggered: %s (query=%r)", reason, query)
        messages = {
            "unsafe_input": "I can't help with that request.",
            "off_topic": "That question looks outside the scope of this knowledge base, so I don't have a reliable answer.",
            "low_retrieval_confidence": "I don't have enough grounded information to answer that confidently.",
            "no_retrieval_results": "I couldn't find anything relevant to answer that.",
            "low_groundedness": "I retrieved some context but couldn't produce an answer that's well-supported by it, so I'm not going to guess.",
            "generation_failed": "Something went wrong generating an answer -- please try again.",
            "stt_failed": "I couldn't transcribe the audio -- please try again.",
        }
        return PipelineResponse(
            query=query,
            transcript=transcript,
            answer=messages.get(reason, "I'm not able to answer that."),
            sources=[],
            confidence=0.0,
            grounded=False,
            guardrail_triggered=reason,
            timings_ms=timings,
        )

    def run_text_query(self, query: str) -> PipelineResponse:
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        ok, reason = checks.check_unsafe_input(query)
        if not ok:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, reason, timings)

        retrieval = self.retriever.retrieve(query)
        timings.update(retrieval.timings_ms)

        # Use the raw ANN similarity captured before hybrid reranking.  The
        # reranked score also contains BM25 and is not comparable to the
        # off-topic threshold.
        top_dense_score = retrieval.timings_ms.get(
            "top_dense_score",
            retrieval.chunks[0].score if retrieval.chunks else 0.0,
        )
        ok, reason = checks.check_off_topic(top_dense_score)
        if not ok:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, reason, timings)

        ok, reason = checks.check_retrieval_confidence(retrieval.chunks)
        if not ok:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, reason, timings)

        try:
            t_gen0 = time.perf_counter()
            generated, gen_ms = self.llm.generate(query, [c.text for c in retrieval.chunks])
            timings["generation_ms"] = gen_ms
        except GenerationError:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "generation_failed", timings)

        ok, reason = checks.check_groundedness(generated.answer, retrieval.chunks)
        if not ok:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, reason, timings)

        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResponse(
            query=query,
            answer=generated.answer,
            sources=[SourceChunk(text=c.text, score=c.score, lang=c.lang, strategy=c.strategy)
                     for c in retrieval.chunks],
            confidence=generated.confidence,
            grounded=True,
            guardrail_triggered=None,
            timings_ms=timings,
        )

    def run_voice_query(self, audio_bytes: bytes, language_code: str | None = None) -> PipelineResponse:
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        try:
            stt_result = self.stt.transcribe(audio_bytes, language_code)
        except STTError:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline("<audio>", "stt_failed", timings)

        timings["stt_ms"] = stt_result["latency_ms"]
        transcript = stt_result["transcript"]

        response = self.run_text_query(transcript)
        response.transcript = transcript
        response.lang_detected = stt_result.get("language_detected")
        response.timings_ms = {**timings, **response.timings_ms}
        return response
