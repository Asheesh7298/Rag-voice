"""
Accuracy evaluation: send the evaluation sets to a deployment and score retrieval, answers and refusals.

    python eval/run_eval.py --url https://...modal.run --label v2-phase1
    python eval/run_eval.py --from-raw eval/runs/eval_v2-phase1.json --label v2-phase1   # re-score only

Sets (built by build_eval_sets.py, except C):
  A  sets/set_a_in_index.jsonl  gold passage is in the index    -> retrieval + answer quality
  B  sets/set_b_heldout.jsonl   validation questions, never indexed -> answer quality and refusals
  C  guardrail cases from queries_parity.jsonl (unsafe, current events, off-topic) -> should be declined

Writes eval/results_eval_<label>.md (report) and eval/results_eval_<label>.json (numbers); the raw
responses go to eval/runs/eval_<label>.json (not committed). Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import unicodedata

from parity_check import ask, git_commit, http_get_json, percentile

LANGS = ("hi", "mr", "en")
STRATEGIES = {"native": "passage_native", "fx": "fixed_overlap", "sem": "semantic_window"}
# Devanagari and Bengali digits -> ASCII; the server's _postprocess converts answer digits the same way.
_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "0123456789" * 2)


# ── Text normalization and answer metrics ───────────────────────────────────

def normalize(text: str) -> list[str]:
    """NFKC, ASCII digits, lower case, punctuation and symbols (Unicode categories P*, S*) replaced by
    spaces, split on whitespace. Vowel signs (category M*) are kept, so Devanagari words stay intact."""
    text = unicodedata.normalize("NFKC", text or "").translate(_DIGITS).lower()
    return "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text).split()


def token_f1(pred: list[str], gold: list[str]) -> float:
    """SQuAD-style F1 over word overlap."""
    common = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    if not pred or not gold or common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def is_subsequence(short: list[str], long: list[str]) -> bool:
    """True if `short` appears as a contiguous run of words inside `long`."""
    n = len(short)
    return n > 0 and any(long[i:i + n] == short for i in range(len(long) - n + 1))


def strategy_of(chunk_id: str | None) -> str:
    suffix = (chunk_id or "").rsplit("-", 1)[-1]
    return next((name for prefix, name in STRATEGIES.items() if suffix.startswith(prefix)), "unknown")


# ── Scoring one question ────────────────────────────────────────────────────

def score(row: dict, resp) -> dict:
    ok = isinstance(resp, dict) and "error" not in resp
    guardrail = resp.get("guardrail_triggered") if ok else "error"
    declined = guardrail is not None
    out = {"id": row["id"], "set": row["set"], "lang": row["lang"], "guardrail": guardrail, "declined": declined}

    if row["set"] in ("A", "B"):
        pred = normalize(resp.get("answer", "")) if not declined else []
        gold = normalize(row["answer"])
        out.update(em=float(bool(pred) and pred == gold), f1=token_f1(pred, gold),
                   contains=float(is_subsequence(pred, gold) or is_subsequence(gold, pred)))

    if row["set"] == "A":
        sources = (resp.get("sources") or []) if ok else []
        gold_passage = row["gold_chunk_id"].rsplit("-", 1)[0]          # e.g. "hi-1234-p0"
        out["rank_passage"] = next((i for i, s in enumerate(sources)
                                    if (s.get("chunk_id") or "").rsplit("-", 1)[0] == gold_passage), None)
        out["rank_query"] = next((i for i, s in enumerate(sources) if s.get("query_id") == row["query_id"]), None)
        out["top1_strategy"] = strategy_of(sources[0].get("chunk_id")) if sources else None
        out["source_strategies"] = [strategy_of(s.get("chunk_id")) for s in sources]

    if row["set"] == "C":
        expect = row["expect"]
        out["expect"] = expect
        out["correct"] = (guardrail == expect) if expect in ("unsafe_input", "out_of_scope") else declined
    return out


# ── Aggregation ─────────────────────────────────────────────────────────────

def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    answered = [r for r in scored if not r["declined"]]
    s = {"n": n, "answered": len(answered), "refusal_rate": 1 - len(answered) / n if n else None}
    if scored and "f1" in scored[0]:
        s.update(em=mean(r["em"] for r in scored), f1=mean(r["f1"] for r in scored),
                 contains=mean(r["contains"] for r in scored),
                 f1_answered=mean(r["f1"] for r in answered), contains_answered=mean(r["contains"] for r in answered))
    if scored and "rank_passage" in scored[0]:
        ranks = [r["rank_passage"] for r in scored]
        s.update(recall_at_1=mean(r == 0 for r in ranks),
                 recall_at_5=mean(r is not None and r < 5 for r in ranks),
                 mrr=mean(1 / (r + 1) if r is not None else 0.0 for r in ranks),
                 recall_at_5_same_query=mean(r["rank_query"] is not None and r["rank_query"] < 5 for r in scored))
    return s


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def build_report(label: str, meta: dict, scored: list[dict], latency: dict) -> tuple[str, dict]:
    by = collections.defaultdict(list)
    for r in scored:
        by[(r["set"], r["lang"])].append(r)
        by[(r["set"], "all")].append(r)
    summary = {f"{k[0]}:{k[1]}": summarize(v) for k, v in by.items() if k[0] in ("A", "B")}

    lines = [f"# Accuracy evaluation: {label}", "",
             f"- Deployment: `{meta['url']}`", f"- Code commit: `{meta.get('git_commit')}`, run {meta['started_at']}",
             f"- Questions: {sum(1 for r in scored if r['set'] == 'A')} in Set A, "
             f"{sum(1 for r in scored if r['set'] == 'B')} in Set B, {sum(1 for r in scored if r['set'] == 'C')} in Set C", ""]

    lines += ["## Set A: the gold passage is in the index", "",
              "| Language | Recall@1 | Recall@5 | MRR | Exact match | F1 | Contains | Refused |",
              "|---|---|---|---|---|---|---|---|"]
    for lang in (*LANGS, "all"):
        s = summary.get(f"A:{lang}")
        if s:
            lines.append(f"| {lang} | {pct(s['recall_at_1'])} | {pct(s['recall_at_5'])} | {s['mrr']:.3f} | "
                         f"{pct(s['em'])} | {pct(s['f1'])} | {pct(s['contains'])} | {pct(s['refusal_rate'])} |")

    lines += ["", "## Set B: held-out questions (never indexed)", "",
              "| Language | Exact match | F1 | Contains | Answered | F1 when answered | Contains when answered |",
              "|---|---|---|---|---|---|---|"]
    for lang in (*LANGS, "all"):
        s = summary.get(f"B:{lang}")
        if s:
            lines.append(f"| {lang} | {pct(s['em'])} | {pct(s['f1'])} | {pct(s['contains'])} | "
                         f"{pct(1 - s['refusal_rate'])} | {pct(s['f1_answered'])} | {pct(s['contains_answered'])} |")

    c_rows = [r for r in scored if r["set"] == "C"]
    c_by = collections.defaultdict(list)
    for r in c_rows:
        c_by[r["expect"]].append(r)
    lines += ["", "## Set C: guardrail cases (should be declined)", "", "| Expected | Correct | Total |", "|---|---|---|"]
    for expect, rows in sorted(c_by.items()):
        lines.append(f"| {expect} | {sum(r['correct'] for r in rows)} | {len(rows)} |")
    summary["C"] = {e: {"correct": sum(r["correct"] for r in rs), "total": len(rs)} for e, rs in c_by.items()}

    a_rows = [r for r in scored if r["set"] == "A"]
    top1_hits = collections.Counter(r["top1_strategy"] for r in a_rows if r["rank_passage"] == 0)
    all_sources = collections.Counter(s for r in a_rows for s in r["source_strategies"])
    lines += ["", "## Chunking strategies (Set A)", "",
              "| Strategy | Top-1 hits on the gold passage | Share of all returned sources |", "|---|---|---|"]
    total_hits, total_sources = sum(top1_hits.values()) or 1, sum(all_sources.values()) or 1
    for name in STRATEGIES.values():
        lines.append(f"| {name} | {top1_hits[name]} ({100 * top1_hits[name] / total_hits:.0f}%) | "
                     f"{100 * all_sources[name] / total_sources:.0f}% |")
    summary["strategies"] = {"top1_hits": dict(top1_hits), "all_sources": dict(all_sources)}

    lines += ["", f"Server latency over all questions: P50 {latency['p50']:.1f} ms, P90 {latency['p90']:.1f} ms.", "",
              "## How to read this", "",
              "- **Recall@k / MRR:** is the gold passage (the exact passage the dataset marks as the answer source) among "
              "the top k sources, and how high (MRR = average of 1/rank). Declined questions return no sources and count as misses.",
              "- **Exact match / F1 / Contains:** the answer compared with the dataset's answer after normalization "
              "(case, punctuation, digits). F1 is word overlap. *Contains* means one answer appears word-for-word inside the other. "
              "Dataset answers are often full sentences while this system extracts short spans, so exact match is naturally low.",
              "- Declined questions score 0 on answer metrics. *When answered* columns exclude them.",
              "- Set A is optimistic: the question's own passage is in the index. Set B shows behavior on unseen questions.",
              "- The strategy table counts which chunking strategy supplied the gold passage when it ranked first; "
              "it is a contribution count, not a controlled ablation."]
    summary["latency_ms"] = latency
    return "\n".join(lines) + "\n", summary


# ── Main ────────────────────────────────────────────────────────────────────

def load_sets() -> list[dict]:
    rows = []
    for path in ("eval/sets/set_a_in_index.jsonl", "eval/sets/set_b_heldout.jsonl"):
        rows += [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for line in open("eval/queries_parity.jsonl", encoding="utf-8"):
        q = json.loads(line)
        if q["category"] in ("unsafe", "current_events", "off_topic"):
            rows.append({**q, "id": f"C-{q['id']}", "set": "C"})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--url", help="deployment base URL")
    p.add_argument("--from-raw", help="re-score a saved raw file instead of querying")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = p.parse_args()

    if args.from_raw:
        raw = json.load(open(args.from_raw, encoding="utf-8"))
    else:
        url = args.url.rstrip("/")
        rows = load_sets()[: args.limit]
        http_get_json(f"{url}/health", timeout=600)
        meta = {"label": args.label, "url": url, "git_commit": git_commit(),
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        results = []
        for i, row in enumerate(rows, 1):
            status, body, client_ms, attempts = ask(url, row["query"])
            results.append({"row": row, "status": status, "attempts": attempts, "client_ms": client_ms, "response": body})
            if i % 25 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
        raw = {"meta": meta, "results": results}
        with open(f"eval/runs/eval_{args.label}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(raw, f, ensure_ascii=False, indent=1)

    scored = [score(r["row"], r["response"]) for r in raw["results"]]
    totals = [r["response"]["timings_ms"]["total_ms"] for r in raw["results"]
              if isinstance(r["response"], dict) and "total_ms" in (r["response"].get("timings_ms") or {})]
    latency = {"p50": percentile(totals, 50), "p90": percentile(totals, 90)}
    report, summary = build_report(args.label, raw["meta"], scored, latency)

    with open(f"eval/results_eval_{args.label}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    with open(f"eval/results_eval_{args.label}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": raw["meta"], "summary": summary}, f, ensure_ascii=False, indent=1)
    print(report)


if __name__ == "__main__":
    main()
