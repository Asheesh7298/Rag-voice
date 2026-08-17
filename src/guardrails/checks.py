"""
Four guardrails, each returning (passed: bool, reason: str | None).
The harness calls these at specific points and short-circuits on failure --
this is the "knows when not to answer" requirement made concrete and testable.
"""
from __future__ import annotations
import re

from src.config import settings
from src.retrieval.retriever import RetrievedChunk

# Minimal unsafe-input keyword list -- deliberately conservative and easy to extend.
# For a real deployment this would be a small classifier; keyword pass is enough to
# demonstrate the guardrail mechanism and catch obvious cases in the demo.
_UNSAFE_PATTERNS = [
    r"\bhow to (make|build) (a )?(bomb|weapon|explosive)\b",
    r"\bself[- ]?harm\b",
    r"\bhack (into|someone)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def check_unsafe_input(query: str) -> tuple[bool, str | None]:
    if _UNSAFE_RE.search(query):
        return False, "unsafe_input"
    return True, None


def check_off_topic(top_score: float) -> tuple[bool, str | None]:
    """top_score = best raw dense similarity score from the initial ANN search,
    taken *before* rerank, as a fast proxy for 'is this query even in-domain'."""
    if top_score < settings.off_topic_threshold:
        return False, "off_topic"
    return True, None


def check_retrieval_confidence(chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
    if not chunks:
        return False, "no_retrieval_results"
    if chunks[0].score < settings.min_retrieval_score:
        return False, "low_retrieval_confidence"
    return True, None


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def check_groundedness(answer: str, sources: list[RetrievedChunk]) -> tuple[bool, str | None]:
    """Cheap lexical-overlap groundedness check: what fraction of the answer's
    content words are traceable to the retrieved context. Not a substitute for an
    NLI entailment model, but fast (no extra model call) and catches the common
    failure mode of the LLM answering from parametric knowledge instead of context.
    Swap in a cross-encoder NLI model here if latency budget allows in later iteration."""
    if not sources:
        return False, "no_sources_to_ground_in"
    combined_context = " ".join(s.text for s in sources)
    overlap = _token_overlap(answer, combined_context)
    if overlap < settings.groundedness_min_overlap:
        return False, "low_groundedness"
    return True, None
