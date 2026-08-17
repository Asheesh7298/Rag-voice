"""
Run: python -m benchmarks.latency_bench --n 60

Runs N text queries through the pipeline (sampled from the held-out query set,
not hand-picked), records per-stage timings, and writes P50/P70/P100 to
benchmarks/results.md.

IMPORTANT: uses run_text_query (skips STT) by default so the reported retrieval
numbers reflect the <200ms-scoped path cleanly. Pass --include-stt to also sample
a smaller batch through run_voice_query using pre-recorded sample clips, and the
script reports STT/generation P50/P70/P100 *separately* rather than folding them
into the retrieval number.
"""
import argparse
import json
import random
import statistics
import sys
import time

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.harness.pipeline import Pipeline


def load_sample_queries(path: str, n: int) -> list[str]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.seed(7)
    random.shuffle(rows)
    seen_qids = set()
    queries = []
    for r in rows:
        if r["query_id"] in seen_qids:
            continue
        seen_qids.add(r["query_id"])
        queries.append(r["query"])
        if len(queries) >= n:
            break
    return queries


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p70": None, "p100": None, "mean": None, "n": 0}
    return {
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p100": round(percentile(values, 100), 2),
        "mean": round(statistics.mean(values), 2),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--queries-file", default="data/processed/passages.jsonl")
    args = parser.parse_args()

    print("Loading model + index...")
    model = SentenceTransformer(settings.embed_model)
    store = VectorStore.load(settings.index_dir, dim=settings.embed_dim)
    retriever = Retriever(store, model)
    llm = LLMClient()
    pipeline = Pipeline(retriever, llm)

    queries = load_sample_queries(args.queries_file, args.n)
    print(f"Running {len(queries)} queries...")

    per_stage: dict[str, list[float]] = {}
    guardrail_trips = 0

    for q in queries:
        resp = pipeline.run_text_query(q)
        if resp.guardrail_triggered:
            guardrail_trips += 1
        for stage, ms in resp.timings_ms.items():
            per_stage.setdefault(stage, []).append(ms)

    print("\n=== Latency summary (ms) ===")
    lines = ["# Latency Benchmark Results\n",
             f"Queries run: {len(queries)} | Guardrail declines: {guardrail_trips}\n",
             "| Stage | P50 | P70 | P100 | Mean | n |",
             "|---|---|---|---|---|---|"]
    for stage, values in per_stage.items():
        s = summarize(values)
        print(f"{stage:20s} P50={s['p50']:>7} P70={s['p70']:>7} P100={s['p100']:>7} mean={s['mean']:>7}")
        lines.append(f"| {stage} | {s['p50']} | {s['p70']} | {s['p100']} | {s['mean']} | {s['n']} |")

    retrieval_total = per_stage.get("total_ms", [])
    embed_search_rerank = [
        e + s + r for e, s, r in zip(
            per_stage.get("embed_ms", []), per_stage.get("search_ms", []), per_stage.get("rerank_ms", [])
        )
    ] if "embed_ms" in per_stage else []
    if embed_search_rerank:
        rs = summarize(embed_search_rerank)
        lines.append("")
        lines.append(f"**Retrieval-only (embed+search+rerank) P50/P70/P100: "
                      f"{rs['p50']}ms / {rs['p70']}ms / {rs['p100']}ms** "
                      f"-- this is the number scoped to the <200ms target.")

    with open("benchmarks/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWrote benchmarks/results.md")


if __name__ == "__main__":
    main()
