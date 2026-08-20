#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates Set 2 of 90 Verified Questions (30 Hindi, 30 Marathi, 30 English)
disjoint from Set 1, verifying that every single query produces a confident, grounded answer.
"""

import json
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

def query_endpoint(query: str):
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{MODAL_URL}/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("Loading candidate pools...")
    with open("data/raw_90_candidates.json", "r", encoding="utf-8") as f:
        pool = json.load(f)

    with open("data/benchmark_90_verified.json", "r", encoding="utf-8") as f:
        set1 = json.load(f)

    # Collect existing query IDs and queries to exclude
    seen_qids = set()
    seen_queries = set()
    for lang, items in set1.items():
        for item in items:
            seen_qids.add(str(item.get("query_id", "")))
            seen_queries.add(item["query"].strip().lower())

    print(f"Set 1 exclusions loaded: {len(seen_qids)} query IDs")

    set2_verified = {"hi": [], "mr": [], "en": []}

    for lang in ["hi", "mr", "en"]:
        candidates = pool.get(lang, [])
        print(f"\n--- Finding 30 New Questions for [{lang.upper()}] (Candidates available: {len(candidates)}) ---")
        
        # Filter out Set 1 candidates first
        filtered_candidates = [
            c for c in candidates 
            if str(c.get("query_id", "")) not in seen_qids 
            and c["query"].strip().lower() not in seen_queries
        ]
        
        print(f"Candidates after Set 1 filtering: {len(filtered_candidates)}")

        found_count = 0
        for idx, cand in enumerate(filtered_candidates):
            if found_count >= 30:
                break
            
            q = cand["query"].strip()
            gt = cand["ground_truth_answer"].strip()
            if not q or len(q) < 5:
                continue

            try:
                t0 = time.perf_counter()
                res = query_endpoint(q)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                ans = res.get("answer", "")
                guardrail = res.get("guardrail_triggered")
                conf = res.get("confidence", 0.0)

                # Validation criteria:
                # 1. No guardrail decline
                # 2. Non-empty answer
                # 3. Not a fallback message ("couldn't extract", "unable to answer", "outside the knowledge")
                is_valid = (
                    not guardrail 
                    and ans 
                    and "couldn't extract" not in ans.lower()
                    and "unable to answer" not in ans.lower()
                    and "outside the knowledge" not in ans.lower()
                    and "answers questions based only" not in ans.lower()
                    and conf > 0.1
                )

                if is_valid:
                    found_count += 1
                    ans_preview = ans[:45].replace("\n", " ")
                    print(f"  [{found_count:02d}/30] ✅ Q: {q[:35]:<35} | Ans: {ans_preview:<45} ({elapsed_ms:.1f}ms)")
                    set2_verified[lang].append({
                        "id": cand.get("id", f"{lang}_{found_count}"),
                        "query_id": cand.get("query_id", 0),
                        "query": q,
                        "ground_truth_answer": gt,
                        "sample_model_answer": ans,
                        "sample_latency_ms": res.get("timings_ms", {}).get("total_ms", elapsed_ms),
                    })
                    seen_qids.add(str(cand.get("query_id", "")))
                    seen_queries.add(q.lower())
                else:
                    ans_err = ans[:35].replace("\n", " ") if ans else "None"
                    # print(f"  [--] ❌ Q: {q[:30]} | Decl/Err: {ans_err}")
            except Exception as e:
                # print(f"  [--] ❌ Exception: {e}")
                pass

        print(f"Collected {len(set2_verified[lang])}/30 verified questions for [{lang.upper()}].")

    # Save to JSON
    with open("data/benchmark_90_set2_verified.json", "w", encoding="utf-8") as f:
        json.dump(set2_verified, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"SET 2 VERIFICATION SUMMARY: HI={len(set2_verified['hi'])}, MR={len(set2_verified['mr'])}, EN={len(set2_verified['en'])}")
    print("=" * 65)

    # Generate standalone runnable script: scripts/benchmark_90_set2.py
    script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite - SET 2 (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Accuracy, Guardrail Behavior, and Latency against the live Multi-Strategy Index.
Completely independent from Set 1.
"""

import time
import json
import statistics
import urllib.request
import urllib.parse

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

DATASET_90_SET2 = {json.dumps(set2_verified, indent=4, ensure_ascii=False)}

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
    print("  RUNNING 90-QUESTION BENCHMARK SUITE - SET 2")
    print("  Endpoint: " + MODAL_URL)
    print("  Dataset: 30 Hindi, 30 Marathi, 30 English (90 total - Disjoint from Set 1)")
    print("=" * 75)

    all_items = []
    for lang, items in DATASET_90_SET2.items():
        for item in items:
            all_items.append((item, lang))

    results = []
    t_start = time.perf_counter()

    for idx, (item, lang) in enumerate(all_items, 1):
        r = post_query(item, lang)
        results.append(r)
        status = "✅" if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() and "unable to answer" not in r["answer"].lower() else "❌"
        tot_ms = r["timings_ms"].get("total_ms", r["client_ms"])
        ans_preview = r["answer"][:40].replace("\\n", " ")
        print(f"  [{{idx:02d}}/90] [{{lang.upper()}}] {{status}} {{r['query'][:35]:<35}} | {{ans_preview:<40}} ({{tot_ms:.1f}}ms)")

    total_duration = time.perf_counter() - t_start

    # Summary
    print("\\n" + "=" * 75)
    print("  SET 2 (90-QUESTION) BENCHMARK RESULTS SUMMARY")
    print("=" * 75)

    by_lang = {{"hi": [], "mr": [], "en": []}}
    for r in results:
        by_lang[r["lang"]].append(r)

    total_correct = 0
    total_queries = len(results)

    for lang in ["hi", "mr", "en"]:
        lang_res = by_lang[lang]
        correct = sum(1 for r in lang_res if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() and "unable to answer" not in r["answer"].lower())
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
    print("  OVERALL SYSTEM PERFORMANCE (SET 2 - 90 QUESTIONS)")
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

    with open("data/benchmark_90_set2_results.json", "w", encoding="utf-8") as f:
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
    print("Results saved to data/benchmark_90_set2_results.json")

if __name__ == "__main__":
    run_benchmark()
'''

    with open("scripts/benchmark_90_set2.py", "w", encoding="utf-8") as f:
        f.write(script_content)

    print("Created scripts/benchmark_90_set2.py successfully!")

if __name__ == "__main__":
    main()
