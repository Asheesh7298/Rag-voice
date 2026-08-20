"""
Measure the exact chunks-per-passage multiplier on 5,000 mixed (hi/mr/en) passages.
"""
import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.25.0",
        "pyarrow>=15.0.0",
        "sentence-transformers>=3.0.1",
        "numpy>=1.26,<3.0",
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "sentencepiece>=0.1.99",
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\""
    )
)

app = modal.App("voice-rag-measure-ratio", image=image)

@app.function(gpu="A10G", timeout=600)
def measure_sample_multiplier(n_sample_per_lang: int = 1666):
    import time, re, random
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    print("Loading SentenceTransformer on CUDA...")
    model = SentenceTransformer("/models/e5-base", device="cuda")
    model.max_seq_length = 64
    model.half()

    def embed_fn(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        with torch.no_grad():
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    # 1. Passage Native
    def chunk_native(p_row):
        t = p_row["text"].strip()
        return [t] if t else []

    # 2. Fixed Overlap (60 tokens, 15 overlap)
    def chunk_fixed(p_row, size=60, overlap=15):
        tokens = p_row["text"].split()
        if not tokens: return []
        chunks = []
        step = max(size - overlap, 1)
        i = 0
        while i < len(tokens):
            w = tokens[i:i + size]
            chunks.append(" ".join(w))
            i += step
            if len(w) < size: break
        return chunks

    # 3. Semantic Window
    sent_split_re = re.compile(r"(?<=[.?!।])\s+")
    def chunk_semantic(p_row, sim_thresh=0.55, max_sents=6):
        text = p_row["text"].strip()
        if not text: return []
        sents = [s.strip() for s in sent_split_re.split(text) if s.strip()]
        if len(sents) <= 1:
            return [text]
        embs = embed_fn(sents)
        chunks = []
        cur = [sents[0]]
        for i in range(1, len(sents)):
            sim = float(np.dot(embs[i-1], embs[i]))
            if sim < sim_thresh or len(cur) >= max_sents:
                chunks.append(" ".join(cur))
                cur = [sents[i]]
            else:
                cur.append(sents[i])
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    # Collect sample passages
    sample_passages = []

    # Hindi
    print("Sampling Hindi passages...")
    pq_path_hi = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="train/hintrain.parquet", repo_type="dataset")
    pf_hi = pq.ParquetFile(pq_path_hi)
    for batch in pf_hi.iter_batches(batch_size=5000, columns=["passages"]):
        for p_struct in batch.to_pydict()["passages"]:
            if isinstance(p_struct, dict):
                for t in p_struct.get("Translated_passages", []):
                    txt = str(t).strip()
                    if len(txt) >= 20:
                        sample_passages.append({"text": txt, "lang": "hi"})
                        if len([p for p in sample_passages if p["lang"] == "hi"]) >= n_sample_per_lang:
                            break
            if len([p for p in sample_passages if p["lang"] == "hi"]) >= n_sample_per_lang:
                break
        if len([p for p in sample_passages if p["lang"] == "hi"]) >= n_sample_per_lang:
            break

    # Marathi
    print("Sampling Marathi passages...")
    pq_path_mr = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="train/martrain.parquet", repo_type="dataset")
    pf_mr = pq.ParquetFile(pq_path_mr)
    for batch in pf_mr.iter_batches(batch_size=5000, columns=["passages"]):
        for p_struct in batch.to_pydict()["passages"]:
            if isinstance(p_struct, dict):
                for t in p_struct.get("Translated_passages", []):
                    txt = str(t).strip()
                    if len(txt) >= 20:
                        sample_passages.append({"text": txt, "lang": "mr"})
                        if len([p for p in sample_passages if p["lang"] == "mr"]) >= n_sample_per_lang:
                            break
            if len([p for p in sample_passages if p["lang"] == "mr"]) >= n_sample_per_lang:
                break
        if len([p for p in sample_passages if p["lang"] == "mr"]) >= n_sample_per_lang:
            break

    # English
    print("Sampling English passages...")
    pq_path_en = hf_hub_download(repo_id="microsoft/ms_marco", filename="v2.1/train-00000-of-00007.parquet", repo_type="dataset")
    pf_en = pq.ParquetFile(pq_path_en)
    for batch in pf_en.iter_batches(batch_size=5000, columns=["passages"]):
        for p_struct in batch.to_pydict()["passages"]:
            if isinstance(p_struct, dict):
                for t in p_struct.get("passage_text", []):
                    txt = str(t).strip()
                    if len(txt) >= 20:
                        sample_passages.append({"text": txt, "lang": "en"})
                        if len([p for p in sample_passages if p["lang"] == "en"]) >= n_sample_per_lang:
                            break
            if len([p for p in sample_passages if p["lang"] == "en"]) >= n_sample_per_lang:
                break
        if len([p for p in sample_passages if p["lang"] == "en"]) >= n_sample_per_lang:
            break

    print(f"\nTotal sampled passages: {len(sample_passages)}")
    by_lang = {}
    for p in sample_passages:
        by_lang[p["lang"]] = by_lang.get(p["lang"], 0) + 1
    print(f"Counts by lang: {by_lang}")

    # Measure chunk counts per strategy
    total_native = 0
    total_fixed = 0
    total_semantic = 0

    t0 = time.perf_counter()
    for idx, p in enumerate(sample_passages):
        c_nat = chunk_native(p)
        c_fix = chunk_fixed(p)
        c_sem = chunk_semantic(p)

        total_native += len(c_nat)
        total_fixed += len(c_fix)
        total_semantic += len(c_sem)

    total_chunks = total_native + total_fixed + total_semantic
    n_passages = len(sample_passages)
    multiplier = total_chunks / n_passages

    print("\n" + "=" * 60)
    print("CHUNK STRATEGY MEASUREMENT REPORT (5,000 sample passages):")
    print(f"Passages evaluated:    {n_passages:,}")
    print(f"Passage native chunks: {total_native:,} (avg {total_native/n_passages:.2f} / passage)")
    print(f"Fixed overlap chunks:  {total_fixed:,} (avg {total_fixed/n_passages:.2f} / passage)")
    print(f"Semantic window chunks:{total_semantic:,} (avg {total_semantic/n_passages:.2f} / passage)")
    print(f"Total chunks produced: {total_chunks:,}")
    print(f"ACTUAL MULTIPLIER:     {multiplier:.4f} chunks / passage")
    print("=" * 60)

    target_total_vectors = 1_500_000
    target_total_passages = int(target_total_vectors / multiplier)
    per_language_passages = target_total_passages // 3

    print(f"\nTarget Total Vectors:   {target_total_vectors:,}")
    print(f"Calculated Total Passages Needed: {target_total_passages:,}")
    print(f"Per-Language Passages Needed:     {per_language_passages:,} (hi, mr, en)")
    print(f"Estimated Final Vectors:          {per_language_passages * 3 * multiplier:,.0f}")
    print("=" * 60)

    return {
        "n_passages": n_passages,
        "multiplier": multiplier,
        "native_avg": total_native / n_passages,
        "fixed_avg": total_fixed / n_passages,
        "semantic_avg": total_semantic / n_passages,
        "target_total_passages": target_total_passages,
        "per_language_passages": per_language_passages,
    }

@app.local_entrypoint()
def main():
    res = measure_sample_multiplier.remote()
    print("Remote measurement result:", res)
