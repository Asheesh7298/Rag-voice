"""
Latency benchmark against the live Modal endpoint.
Run: python benchmarks/modal_bench.py
Measures P50/P70/P100 across 60 real queries from the dataset.
"""
import json, random, statistics, time, sys
import urllib.request
import urllib.parse

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"
N_QUERIES = 60

def percentile(vals, p):
    s = sorted(vals)
    k = (len(s)-1) * p/100
    f, c = int(k), min(int(k)+1, len(s)-1)
    return s[f] + (s[c]-s[f])*(k-f) if f != c else s[f]

def post_query(query):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        f"{MODAL_URL}/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    wall_ms = round((time.perf_counter()-t0)*1000, 2)
    result["wall_ms"] = wall_ms
    return result

# Load real queries from dataset
rows = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]
random.seed(99)
random.shuffle(rows)
# Deduplicate by query_id, pick across all languages
seen = set()
queries = []
for r in rows:
    if r["query_id"] not in seen and r["query"].strip():
        seen.add(r["query_id"])
        queries.append(r["query"])
    if len(queries) >= N_QUERIES:
        break

print(f"Running {len(queries)} queries against {MODAL_URL}...")
print("(first query may be slow if container was idle)\n")

per_stage = {}
guardrail_trips = 0
errors = 0

for i, q in enumerate(queries):
    try:
        result = post_query(q)
        if result.get("guardrail_triggered"):
            guardrail_trips += 1
        for stage, ms in result.get("timings_ms", {}).items():
            per_stage.setdefault(stage, []).append(float(ms))
        per_stage.setdefault("wall_ms", []).append(result["wall_ms"])
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(queries)} done...")
    except Exception as e:
        print(f"  Error on query {i}: {e}")
        errors += 1

print(f"\n=== Latency Results (ms) ===")
print(f"Queries: {len(queries)} | Guardrail declines: {guardrail_trips} | Errors: {errors}")
print(f"{'Stage':25s} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
print("-" * 55)

for stage in ["embed_ms", "search_ms", "rerank_ms", "qa_ms", "total_ms", "wall_ms"]:
    vals = per_stage.get(stage, [])
    if not vals:
        continue
    p50 = round(percentile(vals, 50), 1)
    p70 = round(percentile(vals, 70), 1)
    p100 = round(percentile(vals, 100), 1)
    mean = round(statistics.mean(vals), 1)
    flag = " ✅" if stage == "total_ms" and p100 < 200 else ""
    print(f"{stage:25s} {p50:>8} {p70:>8} {p100:>8} {mean:>8}{flag}")

# Write results.md
lines = [
    "# Latency Benchmark Results\n",
    f"Endpoint: {MODAL_URL}  ",
    f"Queries: {len(queries)} | Guardrail declines: {guardrail_trips} | Errors: {errors}\n",
    "| Stage | P50 | P70 | P100 | Mean |",
    "|---|---|---|---|---|",
]
for stage in ["embed_ms", "search_ms", "rerank_ms", "qa_ms", "total_ms", "wall_ms"]:
    vals = per_stage.get(stage, [])
    if not vals:
        continue
    lines.append(f"| {stage} | {round(percentile(vals,50),1)} | {round(percentile(vals,70),1)} | {round(percentile(vals,100),1)} | {round(statistics.mean(vals),1)} |")

lines += [
    "",
    "## Latency methodology",
    "- `total_ms`: full pipeline excluding STT (embed + search + rerank + QA extraction)",
    "- `wall_ms`: HTTP round-trip from client to Modal endpoint and back",
    "- STT (Sarvam) is a cloud API call (~500-800ms), reported separately",
    "- The <200ms target applies to the retrieval+QA path (total_ms), not end-to-end with STT",
    "- Infrastructure: Modal T4 GPU, min_containers=1 (always warm)",
]
with open("benchmarks/results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nWrote benchmarks/results.md")