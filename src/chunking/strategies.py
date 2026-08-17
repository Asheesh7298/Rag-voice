"""
Three chunking strategies, each producing a stream of Chunk objects tagged with
`chunk_strategy` metadata so retrieval quality can be compared/ablated per strategy.

1. passage_native   - use MSMARCO's own passage boundaries directly. Zero-cost,
                       high precision, but tied to however MSMARCO happened to
                       segment things (sometimes long, sometimes very short).
2. fixed_overlap    - classic fixed-size token window with overlap. Baseline /
                       control group. Robust when passage_native chunks are too
                       long for the embedding model's effective context.
3. semantic_window   - split each passage into sentences, then greedily group
                       consecutive sentences into a chunk until the *embedding
                       distance* between the running chunk and the next sentence
                       exceeds a threshold (a breakpoint). Produces variable-size,
                       topically coherent chunks instead of mid-thought cuts.

Every chunk carries: id, strategy, lang, query_id, source passage id, text,
and (for semantic) the sentence span it covers -- this is the "metadata-aware"
part: retrieval and guardrails can filter/boost by any of these fields later.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    strategy: str
    lang: str
    query_id: str
    source_passage_id: str
    is_selected: bool = False
    extra: dict = field(default_factory=dict)


# ---------- 1. passage-native ----------

def chunk_passage_native(passage_row: dict) -> list[Chunk]:
    text = passage_row["text"].strip()
    if not text:
        return []
    return [Chunk(
        id=f"{passage_row['id']}-native",
        text=text,
        strategy="passage_native",
        lang=passage_row["lang"],
        query_id=passage_row["query_id"],
        source_passage_id=passage_row["id"],
        is_selected=passage_row.get("is_selected", False),
    )]


# ---------- 2. fixed size + overlap ----------

def _simple_tokenize(text: str) -> list[str]:
    # Whitespace tokenization is deliberately script-agnostic (works across all
    # 13 Indic scripts without needing per-language tokenizers).
    return text.split()


def chunk_fixed_overlap(passage_row: dict, size: int = 60, overlap: int = 15) -> list[Chunk]:
    tokens = _simple_tokenize(passage_row["text"])
    if not tokens:
        return []
    chunks = []
    step = max(size - overlap, 1)
    i = 0
    part = 0
    while i < len(tokens):
        window = tokens[i:i + size]
        text = " ".join(window)
        chunks.append(Chunk(
            id=f"{passage_row['id']}-fx{part}",
            text=text,
            strategy="fixed_overlap",
            lang=passage_row["lang"],
            query_id=passage_row["query_id"],
            source_passage_id=passage_row["id"],
            is_selected=passage_row.get("is_selected", False),
            extra={"window_start_tok": i, "window_size": len(window)},
        ))
        part += 1
        i += step
        if len(window) < size:
            break
    return chunks


# ---------- 3. semantic window ----------

_SENT_SPLIT_RE = re.compile(r"(?<=[.?!।])\s+")  # handles Latin punctuation + Hindi danda (।)


def _split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()]


def chunk_semantic(
    passage_row: dict,
    embed_fn,
    similarity_threshold: float = 0.55,
    max_sentences: int = 6,
) -> list[Chunk]:
    """
    embed_fn: callable(list[str]) -> np.ndarray [n, dim], L2-normalized embeddings.
    Greedily grows a chunk while consecutive sentences stay semantically close;
    starts a new chunk on a breakpoint (cosine sim below threshold) or size cap.
    """
    sentences = _split_sentences(passage_row["text"])
    if len(sentences) <= 1:
        # Too short to semantically split -- still tag as semantic_window (not
        # passage_native) so strategy counts/ablations stay clean and don't
        # double-count against the passage_native strategy.
        text = passage_row["text"].strip()
        if not text:
            return []
        return [Chunk(
            id=f"{passage_row['id']}-sem0",
            text=text,
            strategy="semantic_window",
            lang=passage_row["lang"],
            query_id=passage_row["query_id"],
            source_passage_id=passage_row["id"],
            is_selected=passage_row.get("is_selected", False),
            extra={"n_sentences": 1, "single_sentence_fallback": True},
        )]

    embs = embed_fn(sentences)  # [n, dim]
    chunks = []
    current = [sentences[0]]
    part = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i - 1], embs[i]))  # both normalized -> cosine sim
        if sim < similarity_threshold or len(current) >= max_sentences:
            text = " ".join(current)
            chunks.append(Chunk(
                id=f"{passage_row['id']}-sem{part}",
                text=text,
                strategy="semantic_window",
                lang=passage_row["lang"],
                query_id=passage_row["query_id"],
                source_passage_id=passage_row["id"],
                is_selected=passage_row.get("is_selected", False),
                extra={"n_sentences": len(current), "breakpoint_sim": sim},
            ))
            part += 1
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(Chunk(
            id=f"{passage_row['id']}-sem{part}",
            text=" ".join(current),
            strategy="semantic_window",
            lang=passage_row["lang"],
            query_id=passage_row["query_id"],
            source_passage_id=passage_row["id"],
            is_selected=passage_row.get("is_selected", False),
            extra={"n_sentences": len(current)},
        ))
    return chunks


def chunk_all_strategies(passage_row: dict, embed_fn) -> list[Chunk]:
    """Run all three strategies over one passage row. Used by build_index.py."""
    out: list[Chunk] = []
    out += chunk_passage_native(passage_row)
    out += chunk_fixed_overlap(passage_row)
    out += chunk_semantic(passage_row, embed_fn)
    return out
