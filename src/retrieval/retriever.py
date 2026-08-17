"""
The <200ms path: query embed -> FAISS ANN search -> cheap rerank -> context assembly.

Deliberately NO cross-encoder here -- that's usually the single biggest latency
cost in RAG pipelines (50-150ms+ per call on CPU). Instead we do a bi-encoder
similarity rerank we already have for free (the FAISS score itself) combined
with a lightweight BM25 lexical score over the candidate set, which is cheap
(pure Python, no model) and helps catch exact keyword/entity matches that a
dense embedding sometimes misses. This hybrid re-score is a few ms over 20
candidates, not a model forward pass.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.indexing.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    lang: str
    strategy: str
    query_id: str
    chunk_id: str
    is_selected: bool


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    timings_ms: dict  # {"embed": .., "search": .., "rerank": .., "total": ..}


class Retriever:
    def __init__(self, store: VectorStore, embed_model: SentenceTransformer):
        self.store = store
        self.model = embed_model

    def _hybrid_rerank(self, query: str, candidates: list[tuple[dict, float]]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        corpus = [c[0]["text"].split() for c in candidates]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query.split())
        # normalize bm25 scores to [0,1] to combine with cosine sim (already ~[-1,1])
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_scores]

        combined = []
        for (meta, dense_score), bm25_s in zip(candidates, bm25_norm):
            final_score = 0.7 * dense_score + 0.3 * bm25_s
            combined.append(RetrievedChunk(
                text=meta["text"],
                score=final_score,
                lang=meta["lang"],
                strategy=meta["strategy"],
                query_id=meta["query_id"],
                chunk_id=meta["chunk_id"],
                is_selected=meta.get("is_selected", False),
            ))
        combined.sort(key=lambda c: c.score, reverse=True)
        return combined

    def retrieve(self, query: str, top_k: int | None = None, lang_filter: str | None = None) -> RetrievalResult:
        top_k = top_k or settings.top_k
        t0 = time.perf_counter()

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        t1 = time.perf_counter()

        candidates = self.store.search(np.asarray(query_vec), k=settings.rerank_top_n)
        if lang_filter:
            candidates = [c for c in candidates if c[0]["lang"] == lang_filter]
        t2 = time.perf_counter()

        reranked = self._hybrid_rerank(query, candidates)[:top_k]
        t3 = time.perf_counter()

        timings = {
            "embed_ms": round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
        return RetrievalResult(chunks=reranked, timings_ms=timings)
