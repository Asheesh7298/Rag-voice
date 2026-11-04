"""
Build the accuracy-evaluation question sets. Runs on Modal (CPU only) and only *reads* the index Volume.

Set A, "in the index": training questions whose selected (gold) passage is in the index, sampled at random
    from metadata.jsonl. These test retrieval: the right passage exists, so the system should find it.
Set B, "held out": validation questions with a known answer (MSMARCO-XI validation for Hindi/Marathi,
    MS MARCO v2.1 validation for English). The index was built from training data only, so these were never
    seen: the system has to answer from other passages or refuse.

    MODAL_PROFILE=rawrmeinkayanosaurushun modal run eval/build_eval_sets.py

Writes eval/sets/set_a_in_index.jsonl and eval/sets/set_b_heldout.jsonl. Sampling is seeded, so a rerun
picks the same questions.
"""
import ast
import json
import os

import modal

N_PER_LANG = 100
SEED = 42
LANGS = ("hi", "mr", "en")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub", "pyarrow")
volume = modal.Volume.from_name("voice-rag-index")
app = modal.App("voice-rag-eval-sets", image=image)


def clean_answer(value) -> str:
    """Answers are stored as plain strings, lists, or the string form of a list ("['...']")."""
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


@app.function(volumes={"/index": volume}, timeout=1800, cpu=2, memory=4096)
def sample_in_index(n_per_lang: int, seed: int) -> dict:
    import random
    import struct

    rng = random.Random(seed)
    offsets_path = "/index/metadata.offsets"            # int64 byte offset of every metadata line
    n_rows = os.path.getsize(offsets_path) // 8
    picked = {lang: [] for lang in LANGS}
    seen_queries, tries = set(), 0

    with open(offsets_path, "rb") as fo, open("/index/metadata.jsonl", "rb") as fm:
        while any(len(v) < n_per_lang for v in picked.values()) and tries < 200_000:
            tries += 1
            i = rng.randrange(n_rows)
            fo.seek(i * 8)
            fm.seek(struct.unpack("<q", fo.read(8))[0])
            row = json.loads(fm.readline())
            lang, qid = row.get("lang"), row.get("query_id")
            query, answer = (row.get("query") or "").strip(), clean_answer(row.get("answer"))
            if lang not in picked or len(picked[lang]) >= n_per_lang:
                continue
            if not row.get("is_selected") or not query or not answer or qid in seen_queries:
                continue
            seen_queries.add(qid)
            picked[lang].append({
                "id": f"A-{lang}-{len(picked[lang]) + 1:03d}", "set": "A", "lang": lang,
                "query_id": qid, "query": query, "answer": answer, "gold_chunk_id": row.get("chunk_id"),
            })
    return {"rows": [r for lang in LANGS for r in picked[lang]], "tries": tries, "index_rows": n_rows}


@app.function(timeout=1800, cpu=2, memory=16384)
def sample_heldout(n_per_lang: int, seed: int) -> dict:
    import collections
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    specs = [
        ("hi", "ai4bharat/MSMARCO-XI", "validation/hinval.parquet"),
        ("mr", "ai4bharat/MSMARCO-XI", "validation/marval.parquet"),
        ("en", "microsoft/ms_marco", "v2.1/validation-00000-of-00001.parquet"),
    ]
    out, notes = [], {}
    for lang, repo, path in specs:
        local = hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")
        columns = pq.ParquetFile(local).schema_arrow.names
        ans_col = "Answer" if "Answer" in columns else "answers"
        table = pq.read_table(local, columns=["query_id", "query", ans_col]).to_pylist()

        rows = []
        for r in table:
            query, answer = (r["query"] or "").strip(), clean_answer(r[ans_col])
            if query and answer:
                rows.append((r["query_id"], query, answer))
        # Questions without an answer carry a placeholder ("No Answer Present." or its translation);
        # it is by far the most common answer string, so drop it when it covers more than 5% of rows.
        placeholder, count = collections.Counter(a for _, _, a in rows).most_common(1)[0]
        dropped = count > 0.05 * len(rows)
        if dropped:
            rows = [x for x in rows if x[2] != placeholder]
        notes[lang] = {"columns": columns, "rows_with_answer": len(rows),
                       "most_common_answer": placeholder, "count": count, "dropped": dropped}

        rng.shuffle(rows)
        for k, (qid, query, answer) in enumerate(rows[:n_per_lang], 1):
            out.append({"id": f"B-{lang}-{k:03d}", "set": "B", "lang": lang,
                        "query_id": f"{lang}-val-{qid}", "query": query, "answer": answer})
    return {"rows": out, "notes": notes}


@app.local_entrypoint()
def main():
    os.makedirs("eval/sets", exist_ok=True)
    a, b = sample_in_index.remote(N_PER_LANG, SEED), sample_heldout.remote(N_PER_LANG, SEED)
    for name, result in (("set_a_in_index", a), ("set_b_heldout", b)):
        path = f"eval/sets/{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(result['rows'])} questions")
    print("Set A sampling tries:", a["tries"], "of", a["index_rows"], "index rows")
    print("Set B notes:", json.dumps(b["notes"], ensure_ascii=False)[:1500])
