"""
Build the VoxLore index from the source datasets: passages -> three chunking strategies -> e5 embeddings
(ONNX Runtime, the same encoder the server uses) -> the files the server reads. No PyTorch, no FAISS.

Same selection and chunking as the build behind the deployed index (13,020,220 vectors):
  Hindi, Marathi: ai4bharat/MSMARCO-XI train. English: microsoft/ms_marco v2.1 train.
  Per question, keep its selected (gold) passages plus its first passage; drop passages under 20 characters
  and exact duplicates (by text hash, across languages).
  Chunks per passage: passage_native (whole passage), fixed_overlap (60-word windows, step 45),
  semantic_window (groups of up to 4 consecutive sentences).
Rows are written language by language (hi, mr, en); the server relies on that for per-language search.

Output, in a folder on the Volume (default /index/rebuild, never the live files):
  vectors_fp16.bin  [rows, 768] float16, L2-normalized
  metadata.jsonl    one JSON object per row
  metadata.offsets  int64 byte offset of every metadata line

    MODAL_PROFILE=rawrmeinkayanosaurushun modal run scripts/build_index.py                  # full build, GPU
    MODAL_PROFILE=rawrmeinkayanosaurushun modal run scripts/build_index.py --limit 100      # small test, CPU

A test build (--limit) also compares its first rows of each language with the live index.
"""
import os
import sys

import modal

# Same e5 files as the server. The image build cannot import modal_app.py, so the revision is repeated here
# and checked against modal_app.py whenever the script is started locally.
E5_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_URL = f"https://huggingface.co/intfloat/multilingual-e5-base/resolve/{E5_REVISION}"
if modal.is_local():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
    from modal_app import E5_URL as SERVER_E5_URL
    assert SERVER_E5_URL == E5_URL, "build_index.py must use the same e5 revision as modal_app.py"

volume = modal.Volume.from_name("voice-rag-index")
DIM = 768
BATCH = 256


def fetch_e5():
    """Image build step: e5's official ONNX export and tokenizer (same files and revision as the server)."""
    import urllib.request
    os.makedirs("/models/e5", exist_ok=True)
    for local, remote in (("model.onnx", "onnx/model.onnx"), ("tokenizer.json", "tokenizer.json")):
        urllib.request.urlretrieve(f"{E5_URL}/{remote}", f"/models/e5/{local}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.4.6", "tokenizers==0.19.1", "onnxruntime-gpu[cuda,cudnn]==1.31.1",
                 "huggingface_hub==0.36.2", "pyarrow")
    .run_function(fetch_e5)
    .add_local_python_source("backend")
)
app = modal.App("voice-rag-build-index", image=image)


# ── Passages ────────────────────────────────────────────────────────────────

def collect_passages(limit: int = 0):
    """Yield passages in build order (hi, mr, en). `limit` caps passages per language (test builds)."""
    import hashlib

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    seen = set()
    total = 0

    def new_text(text: str) -> bool:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if h in seen:
            return False
        seen.add(h)
        return True

    for lang, filename in (("hi", "train/hintrain.parquet"), ("mr", "train/martrain.parquet")):
        path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=filename, repo_type="dataset")
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["query_id", "passages", "query", "Answer"]):
            d = batch.to_pydict()
            qids, passages, queries, answers = d.get("query_id", []), d.get("passages", []), d.get("query", []), d.get("Answer", [])
            for idx, p_struct in enumerate(passages):
                if not isinstance(p_struct, dict):
                    continue
                qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(count)
                query = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                answer = str(answers[idx]).strip() if idx < len(answers) and answers[idx] else ""
                selected = p_struct.get("is_selected", [])
                for p_idx, p_text in enumerate(p_struct.get("Translated_passages", [])):
                    is_sel = bool(selected[p_idx]) if selected and p_idx < len(selected) else False
                    text = str(p_text).strip()
                    if (not is_sel and p_idx > 0) or len(text) < 20 or not new_text(text):
                        continue
                    count += 1
                    yield {"source_passage_id": f"{lang}-{qid}-p{p_idx}", "text": text, "lang": lang,
                           "query_id": f"{lang}-{qid}", "query": query, "answer": answer, "is_selected": is_sel}
                    if limit and count >= limit:
                        break
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
        total += count

    # English: as many passages as half of Hindi + Marathi, at least 1.1M (as in the original build).
    target = limit or max(total // 2, 1_100_000)
    count = 0
    for shard in range(7):
        if count >= target:
            break
        path = hf_hub_download(repo_id="microsoft/ms_marco", filename=f"v2.1/train-{shard:05d}-of-00007.parquet", repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["query_id", "passages", "query", "answers"]):
            d = batch.to_pydict()
            qids, passages, queries, answers = d.get("query_id", []), d.get("passages", []), d.get("query", []), d.get("answers", [])
            for idx, p_struct in enumerate(passages):
                if not isinstance(p_struct, dict):
                    continue
                qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(count)
                query = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                ans = answers[idx] if idx < len(answers) and answers[idx] else []
                answer = str(ans[0]).strip() if isinstance(ans, list) and ans else ""
                if answer.lower() == "no answer present.":
                    answer = ""
                selected = p_struct.get("is_selected", [])
                for p_idx, p_text in enumerate(p_struct.get("passage_text", [])):
                    is_sel = bool(selected[p_idx]) if selected and p_idx < len(selected) else False
                    text = str(p_text).strip()
                    if (not is_sel and p_idx > 0) or len(text) < 20 or not new_text(text):
                        continue
                    count += 1
                    yield {"source_passage_id": f"en-{qid}-p{p_idx}", "text": text, "lang": "en",
                           "query_id": f"en-{qid}", "query": query, "answer": answer, "is_selected": is_sel}
                    if count >= target:
                        break
                if count >= target:
                    break
            if count >= target:
                break


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk(p: dict) -> list[dict]:
    """The three chunking strategies, in the original order and with the original metadata fields."""
    import re

    text, sp_id = p["text"], p["source_passage_id"]
    base = lambda chunk_id, chunk_text, strategy: {
        "chunk_id": chunk_id, "text": chunk_text, "chunk_strategy": strategy, "lang": p["lang"],
        "query_id": p["query_id"], "source_passage_id": sp_id, "is_selected": p["is_selected"]}
    tail = {"query": p["query"], "answer": p["answer"]}

    chunks = [{**base(f"{sp_id}-native", text, "passage_native"), **tail}]

    tokens = text.split()                                   # fixed_overlap: 60 words, step 45
    i = part = 0
    while i < len(tokens):
        window = tokens[i:i + 60]
        chunks.append({**base(f"{sp_id}-fx{part}", " ".join(window), "fixed_overlap"),
                       "window_start_tok": i, "window_size": len(window), **tail})
        part += 1
        i += 45
        if len(window) < 60:
            break

    sents = [s.strip() for s in re.split(r"(?<=[.?!।])\s+", text) if s.strip()]   # semantic_window
    if len(sents) <= 1:
        groups = [[text]]
    else:
        groups = [[sents[0]]]
        for s in sents[1:]:
            if len(groups[-1]) >= 4:
                groups.append([s])
            else:
                groups[-1].append(s)
    for part, group in enumerate(groups):
        chunks.append({**base(f"{sp_id}-sem{part}", " ".join(group), "semantic_window"),
                       "n_sentences": len(group), **tail})
    return chunks


# ── Build ───────────────────────────────────────────────────────────────────

def _build(limit: int, out: str) -> dict:
    import array
    import json
    import time

    import numpy as np
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()
    from backend.encoders import E5Encoder

    assert os.path.realpath(out) != "/index", "refusing to overwrite the live index files"
    os.makedirs(out, exist_ok=True)
    encoder = E5Encoder("/models/e5", max_length=64)
    t0, rows, pending = time.time(), 0, []
    offsets = array.array("q")
    counts = {}

    with open(f"{out}/vectors_fp16.bin", "wb") as fv, open(f"{out}/metadata.jsonl", "wb") as fm:
        def flush():
            nonlocal rows
            vecs = encoder.embed_batch([c["text"] for c in pending]).astype(np.float16)
            fv.write(vecs.tobytes())
            for c in pending:
                offsets.append(fm.tell())
                fm.write((json.dumps(c, ensure_ascii=False) + "\n").encode("utf-8"))
                counts[c["lang"]] = counts.get(c["lang"], 0) + 1
            rows += len(pending)
            pending.clear()
            if rows % (BATCH * 400) < BATCH:
                print(f"  {rows:,} rows in {time.time() - t0:.0f}s", flush=True)

        for passage in collect_passages(limit):
            pending.extend(chunk(passage))
            while len(pending) >= BATCH:
                rest = pending[BATCH:]
                del pending[BATCH:]
                flush()
                pending.extend(rest)
        if pending:
            flush()

    with open(f"{out}/metadata.offsets", "wb") as fo:
        offsets.tofile(fo)
    volume.commit()
    return {"rows": rows, "rows_per_language": counts, "seconds": round(time.time() - t0)}


def _compare_with_live(out: str) -> dict:
    """Rows of a test build vs the live index: the first rows of each language must be the same chunks
    (same ids, same text) with near-identical vectors (live ones were made by the original PyTorch model)."""
    import json

    import numpy as np

    def reader(folder):
        n = os.path.getsize(f"{folder}/metadata.offsets") // 8
        offs = np.fromfile(f"{folder}/metadata.offsets", dtype=np.int64)
        vecs = np.memmap(f"{folder}/vectors_fp16.bin", dtype=np.float16, mode="r", shape=(n, DIM))
        fm = open(f"{folder}/metadata.jsonl", "rb")

        def row(i):
            fm.seek(int(offs[i]))
            return json.loads(fm.readline()), vecs[i].astype(np.float32)
        return n, row

    n_new, new_row = reader(out)
    n_live, live_row = reader("/index")
    live_start = {}
    for lang in ("hi", "mr", "en"):        # first live row of each language (binary search; blocks are contiguous)
        lo, hi = 0, n_live
        order = {"hi": 0, "mr": 1, "en": 2}
        while lo < hi:
            mid = (lo + hi) // 2
            if order[live_row(mid)[0]["lang"]] < order[lang]:
                lo = mid + 1
            else:
                hi = mid
        live_start[lang] = lo

    seen, same_ids, cosines = {}, 0, []
    for i in range(n_new):
        meta, vec = new_row(i)
        k = seen.get(meta["lang"], 0)
        seen[meta["lang"]] = k + 1
        live_meta, live_vec = live_row(live_start[meta["lang"]] + k)
        same_ids += live_meta["chunk_id"] == meta["chunk_id"] and live_meta["text"] == meta["text"]
        cosines.append(float(vec @ live_vec / (np.linalg.norm(vec) * np.linalg.norm(live_vec))))
    return {"rows_compared": n_new, "same_chunk_and_text": same_ids,
            "cosine_mean": float(np.mean(cosines)), "cosine_min": float(np.min(cosines))}


@app.function(gpu="A10G", volumes={"/index": volume}, timeout=8 * 3600, memory=32768)
def build_gpu(limit: int, out: str) -> dict:
    return _build(limit, out)


@app.function(cpu=4, volumes={"/index": volume}, timeout=3600, memory=16384)
def build_cpu_test(limit: int, out: str) -> dict:
    return {**_build(limit, out), "comparison_with_live": _compare_with_live(out)}


@app.local_entrypoint()
def main(limit: int = 0, out: str = "/index/rebuild"):
    result = build_cpu_test.remote(limit, out) if limit else build_gpu.remote(0, out)
    print(result)
