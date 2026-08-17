"""Measure end-to-end retrieval latency (embed + FAISS search + hybrid rerank) against the
<200ms target budget defined for the local retrieval pipeline.

Usage:
    python benchmarks/benchmark.py [n_queries]
    or
    python -m benchmarks.benchmark [n_queries]
"""
import os
from pathlib import Path
import statistics
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentence_transformers import SentenceTransformer
from src.config import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.retriever import Retriever

LATENCY_BUDGET_MS = 200.0

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Loading embedding model and FAISS vector index...")
    model = SentenceTransformer(settings.embed_model)
    store = VectorStore.load(settings.index_dir, dim=settings.embed_dim)
    retriever = Retriever(store, model)

    print("Warming up (model encode + first index search)...")
    _ = retriever.retrieve("warmup query", top_k=5)

    total_ms, embed_ms, search_ms, rerank_ms = [], [], [], []
    print(f"Benchmarking retrieval pipeline across {n} queries...")
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = retriever.retrieve(query, top_k=5)
        total_ms.append(resp.timings_ms["total_ms"])
        embed_ms.append(resp.timings_ms["embed_ms"])
        search_ms.append(resp.timings_ms["search_ms"])
        rerank_ms.append(resp.timings_ms["rerank_ms"])

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [
        ("embed", embed_ms),
        ("search", search_ms),
        ("rerank", rerank_ms),
        ("total", total_ms),
    ]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget (<200ms)")
    else:
        print(f"FAIL: over budget ({p95_total:.2f}ms > {LATENCY_BUDGET_MS}ms)")
        sys.exit(1)


if __name__ == "__main__":
    main()

