"""
Day 1: pull IndicMSMARCO across all 13 languages, subsample per language to keep
the index fast, and write out a flat JSONL of passages with metadata.

Actual dataset schema (verified against a live sample row): each row IS already
one (query, passage) pair -- no nested passages list. Relevant fields:
  query_id, query, passage (str), passage_id, language, answer (str),
  is_selected (bool), relevance_score (float)

Each output row (one JSONL line per row of the source dataset) looks like:
{
  "id": "hi-10440-p<passage_id or index>",
  "lang": "hi",
  "query_id": "hi-10440",
  "query": "...",
  "text": "<passage text>",
  "is_selected": true/false,
  "answers": ["..."]   # kept as a list for compatibility with downstream code
}
"""
import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "processed", "passages.jsonl")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def row_to_record(row: dict, lang: str, idx: int) -> dict | None:
    text = (row.get("passage") or "").strip()
    if not text:
        return None
    query_id = row.get("query_id") or str(idx)
    passage_id = row.get("passage_id") or str(idx)
    answer = row.get("answer")
    return {
        "id": f"{lang}-{query_id}-p{passage_id}",
        "lang": lang,
        "query_id": f"{lang}-{query_id}",
        "query": row.get("query", ""),
        "text": text,
        "is_selected": bool(row.get("is_selected", False)),
        "answers": [answer] if answer else [],
    }


def main():
    total_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for lang in settings.languages:
            print(f"[{lang}] loading...")
            try:
                ds = load_dataset("ai4bharat/IndicMSMARCO", lang, split="train")
            except Exception as e:
                print(f"  !! failed to load {lang}: {e} -- skipping")
                continue

            n = len(ds)
            idxs = list(range(n))
            random.shuffle(idxs)

            target = settings.max_passages_per_lang
            lang_written = 0
            for i in tqdm(idxs, desc=f"  {lang} rows"):
                if lang_written >= target:
                    break
                row = ds[i]
                record = row_to_record(row, lang, i)
                if record is None:
                    continue
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                lang_written += 1

            print(f"[{lang}] wrote {lang_written} passages (source rows: {n})")
            total_written += lang_written

    print(f"\nDone. Total passages written: {total_written}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()