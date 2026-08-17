from __future__ import annotations
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    text: str
    score: float
    lang: str
    strategy: str


class GeneratedAnswer(BaseModel):
    """Structured output the LLM must produce -- enforced via schema in the prompt
    and validated on parse. If validation fails, the harness retries with a
    corrective follow-up before falling back to a safe decline."""
    answer: str
    used_source_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PipelineResponse(BaseModel):
    query: str
    transcript: str | None = None
    answer: str
    sources: list[SourceChunk]
    confidence: float
    grounded: bool
    guardrail_triggered: str | None = None  # None if no guardrail fired
    timings_ms: dict[str, float]
    lang_detected: str | None = None
