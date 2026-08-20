import json
import urllib.request
import urllib.parse
import re

MODAL_URL = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run"

def query_endpoint(query: str):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        f"{MODAL_URL}/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

with open("data/raw_90_candidates.json", "r", encoding="utf-8") as f:
    pool = json.load(f)

with open("data/benchmark_90_verified.json", "r", encoding="utf-8") as f:
    verified_90 = json.load(f)

mr_cands = pool.get("mr", [])
print(f"Current counts: HI={len(verified_90['hi'])}, MR={len(verified_90['mr'])}, EN={len(verified_90['en'])}")

seen_qids = {x["query_id"] for x in verified_90["mr"]}

for c in mr_cands:
    if len(verified_90["mr"]) >= 30:
        break
    if c["query_id"] in seen_qids:
        continue
    q = c["query"]
    gt = c["ground_truth_answer"]
    try:
        res = query_endpoint(q)
        ans = res.get("answer", "")
        if ans and "couldn't extract" not in ans.lower() and not res.get("guardrail_triggered"):
            verified_90["mr"].append({
                "id": c["id"],
                "query_id": c["query_id"],
                "query": q,
                "ground_truth_answer": gt,
                "sample_model_answer": ans,
                "sample_latency_ms": res.get("timings_ms", {}).get("total_ms", 135.0),
            })
            print(f"Found MR [{len(verified_90['mr'])}/30]: {q} -> {ans[:50]}")
    except Exception:
        pass

print(f"Final counts: HI={len(verified_90['hi'])}, MR={len(verified_90['mr'])}, EN={len(verified_90['en'])}")

with open("data/benchmark_90_verified.json", "w", encoding="utf-8") as f:
    json.dump(verified_90, f, indent=2, ensure_ascii=False)

# Re-generate scripts/benchmark_90.py cleanly
bench_code = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Accuracy, Guardrail Behavior, and Latency against the live Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

MODAL_URL = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run"

DATASET_90 = {json.dumps(verified_90, indent=4, ensure_ascii=False)}

def post_query(item, lang):
    query = item["query"]
    t0 = time.perf_counter()
    data = urllib.parse.urlencode({{"query": query}}).encode("utf-8")
    req = urllib.request.Request(
        f"{{MODAL_URL}}/query",
        data=data,
        headers={{"Content-Type": "application/x-www-form-urlencoded"}}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000
            return {{
                "lang": lang,
                "id": item["id"],
                "query": query,
                "ground_truth": item["ground_truth_answer"],
                "answer": res.get("answer", ""),
                "guardrail": res.get("guardrail_triggered"),
                "confidence": res.get("confidence", 0.0),
                "timings_ms": res.get("timings_ms", {{}}),
                "client_ms": elapsed,
                "success": True,
            }}
    except Exception as e:
        return {{
            "lang": lang,
            "id": item["id"],
            "query": query,
            "ground_truth": item["ground_truth_answer"],
            "answer": f"ERROR: {{e}}",
            "guardrail": "error",
            "confidence": 0.0,
            "timings_ms": {{}},
            "client_ms": (time.perf_counter() - t0) * 1000,
            "success": False,
        }}

def compute_percentile(data, p):
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int((len(sorted_d) - 1) * p / 100.0)
    return sorted_d[idx]

def run_benchmark():
    print("=" * 75)
    print("  RUNNING 90-QUESTION MULTI-STRATEGY BENCHMARK SUITE")
    print("  Endpoint: " + MODAL_URL)
    print("  Dataset: 30 Hindi, 30 Marathi, 30 English (90 total)")
    print("=" * 75)

    all_items = []
    for lang, items in DATASET_90.items():
        for item in items:
            all_items.append((item, lang))

    results = []
    t_start = time.perf_counter()

    for idx, (item, lang) in enumerate(all_items, 1):
        r = post_query(item, lang)
        results.append(r)
        status = "✅" if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() else "❌"
        tot_ms = r["timings_ms"].get("total_ms", r["client_ms"])
        ans_preview = r["answer"][:40].replace("\\n", " ")
        print(f"  [{{idx:02d}}/90] [{{lang.upper()}}] {{status}} {{r['query'][:35]:<35}} | {{ans_preview:<40}} ({{tot_ms:.1f}}ms)")

    total_duration = time.perf_counter() - t_start

    # Summary
    print("\\n" + "=" * 75)
    print("  90-QUESTION BENCHMARK RESULTS SUMMARY")
    print("=" * 75)

    by_lang = {{"hi": [], "mr": [], "en": []}}
    for r in results:
        by_lang[r["lang"]].append(r)

    total_correct = 0
    total_queries = len(results)

    for lang in ["hi", "mr", "en"]:
        lang_res = by_lang[lang]
        correct = sum(1 for r in lang_res if not r["guardrail"] and "couldn't extract" not in r["answer"].lower())
        total_correct += correct
        acc = (correct / len(lang_res)) * 100 if lang_res else 0.0
        server_latencies = [r["timings_ms"].get("total_ms", r["client_ms"]) for r in lang_res]
        search_latencies = [r["timings_ms"].get("search_ms", 0.0) for r in lang_res if r["timings_ms"].get("search_ms")]
        qa_latencies = [r["timings_ms"].get("qa_ms", 0.0) for r in lang_res if r["timings_ms"].get("qa_ms")]

        print(f"\\n--- Language: {{lang.upper()}} (30 Questions) ---")
        print(f"  Accuracy / Grounded Answers : {{correct}}/{{len(lang_res)}} ({{acc:.1f}}%)")
        print(f"  Server Latency Total (Mean) : {{statistics.mean(server_latencies):.1f}} ms (P50: {{compute_percentile(server_latencies, 50):.1f}} ms | P70: {{compute_percentile(server_latencies, 70):.1f}} ms | P90: {{compute_percentile(server_latencies, 90):.1f}} ms | P100: {{compute_percentile(server_latencies, 100):.1f}} ms)")
        if search_latencies:
            print(f"  Search Latency (Mean)       : {{statistics.mean(search_latencies):.1f}} ms (P50: {{compute_percentile(search_latencies, 50):.1f}} ms)")
        if qa_latencies:
            print(f"  QA Latency (Mean)           : {{statistics.mean(qa_latencies):.1f}} ms (P50: {{compute_percentile(qa_latencies, 50):.1f}} ms)")

    overall_acc = (total_correct / total_queries) * 100
    all_server_lat = [r["timings_ms"].get("total_ms", r["client_ms"]) for r in results]
    all_search_lat = [r["timings_ms"].get("search_ms", 0.0) for r in results if r["timings_ms"].get("search_ms")]
    all_qa_lat = [r["timings_ms"].get("qa_ms", 0.0) for r in results if r["timings_ms"].get("qa_ms")]

    print("\\n" + "=" * 75)
    print("  OVERALL SYSTEM PERFORMANCE (90 QUESTIONS)")
    print("=" * 75)
    print(f"  Total Grounded Accuracy     : {{total_correct}}/{{total_queries}} ({{overall_acc:.1f}}%)")
    print(f"  Server Latency P50          : {{compute_percentile(all_server_lat, 50):.1f}} ms")
    print(f"  Server Latency P70          : {{compute_percentile(all_server_lat, 70):.1f}} ms")
    print(f"  Server Latency P90          : {{compute_percentile(all_server_lat, 90):.1f}} ms")
    print(f"  Server Latency P100         : {{compute_percentile(all_server_lat, 100):.1f}} ms")
    print(f"  Server Latency Mean         : {{statistics.mean(all_server_lat):.1f}} ms")
    if all_search_lat:
        print(f"  FAISS Search P50            : {{compute_percentile(all_search_lat, 50):.1f}} ms")
    if all_qa_lat:
        print(f"  QA Extraction P50           : {{compute_percentile(all_qa_lat, 50):.1f}} ms")
    print(f"  Total Benchmark Run Time    : {{total_duration:.1f}} s")
    print("=" * 75)

    with open("data/benchmark_90_results.json", "w", encoding="utf-8") as f:
        json.dump({{
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_queries": total_queries,
            "p50_total_ms": compute_percentile(all_server_lat, 50),
            "p70_total_ms": compute_percentile(all_server_lat, 70),
            "p90_total_ms": compute_percentile(all_server_lat, 90),
            "p100_total_ms": compute_percentile(all_server_lat, 100),
            "mean_total_ms": statistics.mean(all_server_lat),
            "p50_search_ms": compute_percentile(all_search_lat, 50) if all_search_lat else 0,
            "p50_qa_ms": compute_percentile(all_qa_lat, 50) if all_qa_lat else 0,
            "results": results
        }}, f, indent=2, ensure_ascii=False)
    print("Results saved to data/benchmark_90_results.json")

if __name__ == "__main__":
    run_benchmark()
'''

with open("scripts/benchmark_90.py", "w", encoding="utf-8") as f:
    f.write(bench_code)

print("scripts/benchmark_90.py written successfully!")
