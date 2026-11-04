"""
Parity check: capture full /query responses from a deployment, then compare two captures.

The migration rule is "same answers, no slower": this tool is how we prove it.

Capture (sends every question in the query file, one at a time, like the frontend does):
    python eval/parity_check.py capture --url https://...modal.run --out eval/runs/v1_run1.json --label v1-run1

Compare two captures (answers, guardrail decisions, sources, latency):
    python eval/parity_check.py compare eval/runs/v1_run1.json eval/runs/v1_run2.json --report eval/runs/v1_noise.md

Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGES = ("embed_ms", "search_ms", "rerank_ms", "qa_ms", "total_ms")


# ── Capture ─────────────────────────────────────────────────────────────────

def http_get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_query(url: str, query: str, timeout: float) -> tuple[int, dict | str, float]:
    """POST the query as form data (same as the frontend). Returns (status, body, round_trip_ms)."""
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(f"{url}/query", data=data, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode("utf-8", errors="replace")
    ms = round((time.perf_counter() - t0) * 1000, 1)
    try:
        return status, json.loads(raw), ms
    except json.JSONDecodeError:
        return status, raw, ms


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def capture(args) -> None:
    url = args.url.rstrip("/")
    queries = [json.loads(line) for line in open(args.queries, encoding="utf-8") if line.strip()]

    print(f"Waiting for {url}/health ...")
    t0 = time.perf_counter()
    health = http_get_json(f"{url}/health", timeout=600)
    print(f"  ready in {time.perf_counter() - t0:.1f}s: index_size={health.get('index_size')}")

    # One unrecorded query so the first measured question isn't paying any leftover warm-up cost.
    post_query(url, "what county is columbus city in", timeout=120)

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    results = []
    for i, q in enumerate(queries, 1):
        # Retry Modal infrastructure failures (e.g. GPU preemption returns a non-JSON 5xx while a new
        # container boots, ~35s) so they don't show up as answer differences.
        for attempt in range(1, 5):
            status, body, client_ms = post_query(url, q["query"], timeout=120)
            if status < 500 and isinstance(body, dict):
                break
            print(f"  {q['id']}: attempt {attempt} got HTTP {status}, retrying in {15 * attempt}s")
            time.sleep(15 * attempt)
        results.append({**q, "status": status, "attempts": attempt, "client_ms": client_ms, "response": body})
        answer = body.get("answer") if isinstance(body, dict) else body
        server = body.get("timings_ms", {}).get("total_ms") if isinstance(body, dict) else None
        print(f"  [{i:03d}/{len(queries)}] {q['id']:<14} {status} {server!s:>8}ms  {str(answer)[:60]!r}")

    out = {
        "meta": {
            "label": args.label,
            "url": url,
            "git_commit": git_commit(),
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_queries": len(results),
            "health": health,
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(results)} responses -> {args.out}")


# ── Compare ─────────────────────────────────────────────────────────────────

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def source_keys(resp: dict) -> list[str]:
    """Identify sources by chunk_id when the server provides it, otherwise by their text."""
    return [s.get("chunk_id") or s.get("text", "") for s in resp.get("sources") or []]


def latency_table(results: list[dict]) -> dict[str, dict[str, float | None]]:
    table = {}
    for stage in STAGES:
        vals = [r["response"]["timings_ms"][stage] for r in results
                if isinstance(r["response"], dict) and stage in (r["response"].get("timings_ms") or {})]
        table[stage] = {"n": len(vals), "p50": percentile(vals, 50), "p90": percentile(vals, 90),
                        "max": max(vals) if vals else None}
    client = [r["client_ms"] for r in results]
    table["client_round_trip_ms"] = {"n": len(client), "p50": percentile(client, 50),
                                     "p90": percentile(client, 90), "max": max(client) if client else None}
    return table


def compare(args) -> None:
    a, b = (json.load(open(p, encoding="utf-8")) for p in (args.a, args.b))
    rb = {r["id"]: r for r in b["results"]}
    rows = [(ra, rb[ra["id"]]) for ra in a["results"] if ra["id"] in rb]

    counts = {"answer": 0, "guardrail": 0, "top1": 0, "top8": 0, "status": 0}
    diffs = []
    for ra, rbb in rows:
        x, y = ra["response"], rbb["response"]
        if not (isinstance(x, dict) and isinstance(y, dict)):
            same = x == y
            for k in counts:
                counts[k] += same
            if not same:
                diffs.append((ra, rbb, ["raw response differs"]))
            continue
        kx, ky = source_keys(x), source_keys(y)
        checks = {
            "status": ra["status"] == rbb["status"],
            "answer": x.get("answer") == y.get("answer"),
            "guardrail": x.get("guardrail_triggered") == y.get("guardrail_triggered"),
            "top1": kx[:1] == ky[:1],
            "top8": set(kx[:8]) == set(ky[:8]),
        }
        for k, ok in checks.items():
            counts[k] += ok
        if not all(checks.values()):
            diffs.append((ra, rbb, [k for k, ok in checks.items() if not ok]))

    n = len(rows)
    lines = [
        f"# Parity: {a['meta']['label']} vs {b['meta']['label']}",
        "",
        f"- A: `{args.a}` (commit {a['meta'].get('git_commit')}, {a['meta']['started_at']})",
        f"- B: `{args.b}` (commit {b['meta'].get('git_commit')}, {b['meta']['started_at']})",
        f"- Questions compared: {n}",
        "",
        "| Check | Identical | % |",
        "|---|---|---|",
    ]
    for k, label in (("answer", "Answer text"), ("guardrail", "Guardrail decision"),
                     ("top1", "Top source"), ("top8", "Top-8 sources (as a set)"), ("status", "HTTP status")):
        lines.append(f"| {label} | {counts[k]}/{n} | {100 * counts[k] / n:.1f}% |")

    la, lb = latency_table(a["results"]), latency_table(b["results"])
    lines += ["", "## Latency (ms)", "", "| Stage | A P50 | B P50 | A P90 | B P90 | A max | B max |",
              "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.1f}"
    for stage in (*STAGES, "client_round_trip_ms"):
        sa, sb = la[stage], lb[stage]
        lines.append(f"| {stage} | {fmt(sa['p50'])} | {fmt(sb['p50'])} | {fmt(sa['p90'])} | "
                     f"{fmt(sb['p90'])} | {fmt(sa['max'])} | {fmt(sb['max'])} |")
    lines += ["", "Server stages come from the response's `timings_ms` (questions stopped early by a guardrail "
              "have fewer stages). `client_round_trip_ms` also includes the network to Modal.", ""]

    lines += [f"## Differences ({len(diffs)})", ""]
    for ra, rbb, failed in diffs:
        x, y = ra["response"], rbb["response"]
        ga = x.get("guardrail_triggered") if isinstance(x, dict) else None
        gb = y.get("guardrail_triggered") if isinstance(y, dict) else None
        lines += [
            f"### {ra['id']} ({ra['category']}, {ra['lang']}): differs in {', '.join(failed)}",
            f"- Query: {ra['query'][:200]!r}",
            f"- A: {(x.get('answer') if isinstance(x, dict) else x)!r} (guardrail={ga})",
            f"- B: {(y.get('answer') if isinstance(y, dict) else y)!r} (guardrail={gb})",
            "",
        ]

    report = "\n".join(lines)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as f:
            f.write(report + "\n")
        print(f"Report -> {args.report}")
    print(report)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="send every question and save full responses")
    c.add_argument("--url", required=True, help="deployment base URL (no trailing /query)")
    c.add_argument("--out", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--queries", default="eval/queries_parity.jsonl")
    c.set_defaults(func=capture)

    d = sub.add_parser("compare", help="compare two captures")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--report", help="also write the report to this markdown file")
    d.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
