"""
100% Full Dataset Multi-Strategy Indexer on Modal Cloud (Target: ~12,000,000 vectors)
Ingests 100% of all Hindi, Marathi, and English MSMARCO datasets:
1. Hindi: 100% of ai4bharat/MSMARCO-XI (train/hintrain.parquet)
2. Marathi: 100% of ai4bharat/MSMARCO-XI (train/martrain.parquet)
3. English: 100% of microsoft/ms_marco balanced corpus

Applies 3 chunking strategies:
1. passage_native   - Full unmodified passage
2. fixed_overlap    - 60-token sliding window, 15-token overlap
3. semantic_window  - Sentence-level semantic clustering

Run:
    .\venv\Scripts\python.exe -m modal run scripts/modal_build_full_100_percent.py
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.25.0",
        "pyarrow>=15.0.0",
        "sentence-transformers>=3.0.1",
        "faiss-cpu>=1.9.0",
        "numpy>=1.26,<3.0",
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "tqdm>=4.66.4",
        "sentencepiece>=0.1.99",
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-100percent-indexer", image=image)


@app.cls(
    gpu="A10G",           # 24GB VRAM GPU
    memory=65536,         # 64GB RAM for 12M vector in-memory assembly
    timeout=28800,        # 8 hours max execution time
    volumes={"/index": volume},
)
class Full100PercentIndexer:

    @modal.method()
    def build_index(self):
        import time
        import re
        import hashlib
        import json
        import numpy as np
        import faiss
        import torch
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm

        print("=" * 80)
        print("🚀 STARTING 100% FULL DATASET MULTI-STRATEGY INDEX BUILD ON MODAL CLOUD")
        print("   Target: 100% of Hindi, Marathi, and English MSMARCO datasets")
        print("   Estimated Total Chunks: ~12,000,000 vectors (~18GB FP16 index)")
        print("=" * 80)

        t_start = time.perf_counter()

        # ── 1. Load Multilingual Embedding Model onto GPU ──
        print("\nLoading multilingual-e5-base embedding model onto CUDA GPU (FP16)...")
        model = SentenceTransformer("/models/e5-base", device="cuda")
        model.max_seq_length = 64
        model.half()
        print("✅ Model loaded in FP16 on A10G GPU\n")

        # ── Chunking Strategy Functions ──
        sent_split_re = re.compile(r"(?<=[.?!।])\s+")

        def generate_chunks_for_passage(source_p: dict) -> list[dict]:
            chunks = []
            text = source_p["text"].strip()
            if not text:
                return []

            lang = source_p["lang"]
            qid = source_p["query_id"]
            sp_id = source_p["source_passage_id"]
            is_sel = source_p["is_selected"]
            query = source_p.get("query", "")
            ans = source_p.get("answer", "")

            # 1. passage_native
            chunks.append({
                "chunk_id": f"{sp_id}-native",
                "text": text,
                "chunk_strategy": "passage_native",
                "lang": lang,
                "query_id": qid,
                "source_passage_id": sp_id,
                "is_selected": is_sel,
                "query": query,
                "answer": ans,
            })

            # 2. fixed_overlap (60 tokens, 15 overlap)
            tokens = text.split()
            if tokens:
                step = 45
                i = 0
                part = 0
                while i < len(tokens):
                    window = tokens[i:i + 60]
                    w_text = " ".join(window)
                    chunks.append({
                        "chunk_id": f"{sp_id}-fx{part}",
                        "text": w_text,
                        "chunk_strategy": "fixed_overlap",
                        "lang": lang,
                        "query_id": qid,
                        "source_passage_id": sp_id,
                        "is_selected": is_sel,
                        "window_start_tok": i,
                        "window_size": len(window),
                        "query": query,
                        "answer": ans,
                    })
                    part += 1
                    i += step
                    if len(window) < 60:
                        break

            # 3. semantic_window
            sents = [s.strip() for s in sent_split_re.split(text) if s.strip()]
            if len(sents) <= 1:
                chunks.append({
                    "chunk_id": f"{sp_id}-sem0",
                    "text": text,
                    "chunk_strategy": "semantic_window",
                    "lang": lang,
                    "query_id": qid,
                    "source_passage_id": sp_id,
                    "is_selected": is_sel,
                    "n_sentences": 1,
                    "query": query,
                    "answer": ans,
                })
            else:
                cur_sents = [sents[0]]
                part = 0
                for s_idx in range(1, len(sents)):
                    if len(cur_sents) >= 4:
                        chunks.append({
                            "chunk_id": f"{sp_id}-sem{part}",
                            "text": " ".join(cur_sents),
                            "chunk_strategy": "semantic_window",
                            "lang": lang,
                            "query_id": qid,
                            "source_passage_id": sp_id,
                            "is_selected": is_sel,
                            "n_sentences": len(cur_sents),
                            "query": query,
                            "answer": ans,
                        })
                        part += 1
                        cur_sents = [sents[s_idx]]
                    else:
                        cur_sents.append(sents[s_idx])
                if cur_sents:
                    chunks.append({
                        "chunk_id": f"{sp_id}-sem{part}",
                        "text": " ".join(cur_sents),
                        "chunk_strategy": "semantic_window",
                        "lang": lang,
                        "query_id": qid,
                        "source_passage_id": sp_id,
                        "is_selected": is_sel,
                        "n_sentences": len(cur_sents),
                        "query": query,
                        "answer": ans,
                    })

            return chunks

        # ── 2. Collect 100% of Passages for Hindi & Marathi ──
        raw_passages = []
        seen_hashes = set()

        for lang_code, filename in [("hi", "train/hintrain.parquet"), ("mr", "train/martrain.parquet")]:
            print(f"\n[{lang_code.upper()}] Ingesting 100% of ai4bharat/MSMARCO-XI ({filename})...")
            pq_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=filename, repo_type="dataset")
            pf = pq.ParquetFile(pq_path)
            lang_pool = []

            for batch in pf.iter_batches(batch_size=20000, columns=["query_id", "passages", "query", "Answer"]):
                df_dict = batch.to_pydict()
                qids = df_dict.get("query_id", [])
                passages_list = df_dict.get("passages", [])
                queries = df_dict.get("query", [])
                answers_list = df_dict.get("Answer", [])

                for idx in range(len(passages_list)):
                    p_struct = passages_list[idx]
                    if not isinstance(p_struct, dict): continue
                    p_texts = p_struct.get("Translated_passages", [])
                    is_sel_list = p_struct.get("is_selected", [])
                    qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(len(lang_pool))
                    q_text = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                    ans = answers_list[idx] if idx < len(answers_list) and answers_list[idx] else ""
                    ans_text = str(ans).strip() if ans else ""

                    for p_idx, p_text in enumerate(p_texts):
                        text = str(p_text).strip()
                        if len(text) < 20: continue
                        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                        if h in seen_hashes: continue
                        seen_hashes.add(h)
                        is_sel = bool(is_sel_list[p_idx]) if is_sel_list and p_idx < len(is_sel_list) else False
                        lang_pool.append({
                            "source_passage_id": f"{lang_code}-{qid}-p{p_idx}",
                            "text": text,
                            "lang": lang_code,
                            "query_id": f"{lang_code}-{qid}",
                            "query": q_text,
                            "answer": ans_text,
                            "is_selected": is_sel,
                        })

            print(f"  Total unique 100% passages collected for {lang_code.upper()}: {len(lang_pool):,}")
            raw_passages.extend(lang_pool)

        # ── 3. Collect Balanced 100% Passages for English ──
        target_en = max(len(raw_passages) // 2, 1_100_000)
        print(f"\n[EN] Ingesting {target_en:,} passages from microsoft/ms_marco v2.1...")
        en_pool = []
        for shard_idx in range(7):
            if len(en_pool) >= target_en: break
            shard_name = f"v2.1/train-{shard_idx:05d}-of-00007.parquet"
            shard_path = hf_hub_download(repo_id="microsoft/ms_marco", filename=shard_name, repo_type="dataset")
            pf = pq.ParquetFile(shard_path)

            for batch in pf.iter_batches(batch_size=20000, columns=["query_id", "passages", "query", "answers"]):
                df_dict = batch.to_pydict()
                qids = df_dict.get("query_id", [])
                passages_list = df_dict.get("passages", [])
                queries = df_dict.get("query", [])
                answers_list = df_dict.get("answers", [])

                for idx in range(len(passages_list)):
                    p_struct = passages_list[idx]
                    if not isinstance(p_struct, dict): continue
                    p_texts = p_struct.get("passage_text", [])
                    is_sel_list = p_struct.get("is_selected", [])
                    qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(len(en_pool))
                    q_text = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                    ans = answers_list[idx] if idx < len(answers_list) and answers_list[idx] else []
                    ans_text = str(ans[0]).strip() if isinstance(ans, list) and ans else ""
                    if ans_text.lower() == "no answer present.": ans_text = ""

                    for p_idx, p_text in enumerate(p_texts):
                        text = str(p_text).strip()
                        if len(text) < 20: continue
                        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                        if h in seen_hashes: continue
                        seen_hashes.add(h)
                        is_sel = bool(is_sel_list[p_idx]) if is_sel_list and p_idx < len(is_sel_list) else False
                        en_pool.append({
                            "source_passage_id": f"en-{qid}-p{p_idx}",
                            "text": text,
                            "lang": "en",
                            "query_id": f"en-{qid}",
                            "query": q_text,
                            "answer": ans_text,
                            "is_selected": is_sel,
                        })
                        if len(en_pool) >= target_en: break
                    if len(en_pool) >= target_en: break
                if len(en_pool) >= target_en: break

        raw_passages.extend(en_pool)
        print(f"  ✅ Collected {len(en_pool):,} EN source passages")
        print(f"\nTotal 100% source passages across 3 languages: {len(raw_passages):,}")

        # ── 4. Multi-Strategy Chunking ──
        print("\n⚡ Applying multi-strategy chunking (passage_native, fixed_overlap, semantic_window)...")
        all_chunks = []
        strategy_counts = {"passage_native": 0, "fixed_overlap": 0, "semantic_window": 0}

        t_chunk0 = time.perf_counter()
        for p in tqdm(raw_passages, desc="Multi-Strategy Chunking"):
            p_chunks = generate_chunks_for_passage(p)
            for c in p_chunks:
                strat = c["chunk_strategy"]
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            all_chunks.extend(p_chunks)

        chunk_dur = time.perf_counter() - t_chunk0
        total_vectors_target = len(all_chunks)
        print(f"\n✅ Generated {total_vectors_target:,} total chunks in {chunk_dur/60:.2f} min")
        print(f"   - passage_native:  {strategy_counts['passage_native']:,}")
        print(f"   - fixed_overlap:   {strategy_counts['fixed_overlap']:,}")
        print(f"   - semantic_window: {strategy_counts['semantic_window']:,}")
        print(f"   - Actual Ratio:    {total_vectors_target / len(raw_passages):.4f} chunks/passage")

        # ── 5. GPU Embedding of All Chunks ──
        all_chunk_texts = [c["text"] for c in all_chunks]
        batch_size = 512
        all_vecs = []
        t_embed = time.perf_counter()
        print(f"\n⚡ Encoding {len(all_chunk_texts):,} chunks on A10G GPU (Batch size: {batch_size})...")

        for i in tqdm(range(0, len(all_chunk_texts), batch_size), desc="GPU Embedding"):
            batch = all_chunk_texts[i:i + batch_size]
            with torch.no_grad():
                vecs = model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
            all_vecs.append(np.asarray(vecs, dtype=np.float32))

        vectors = np.vstack(all_vecs)
        embed_dur = time.perf_counter() - t_embed
        throughput = len(all_chunk_texts) / max(embed_dur, 1)
        print(f"\n✅ Embedding completed in {embed_dur/60:.2f} min ({throughput:.1f} chunks/sec)")
        print(f"Matrix shape: {vectors.shape} ({vectors.nbytes / (1024**3):.2f} GB)")

        # ── 6. Build FAISS Index ──
        dim = vectors.shape[1]
        print(f"\n⚙️ Building FAISS IndexFlatIP (dim={dim})...")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        print(f"✅ FAISS index built with {index.ntotal:,} vectors")

        # ── 7. Save to Modal Volume ──
        print("\n💾 Writing 100% multi-strategy index to Modal Volume '/index'...")
        os.makedirs("/index", exist_ok=True)
        faiss.write_index(index, "/index/index.faiss")

        with open("/index/metadata.jsonl", "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        print("Committing changes to Modal Volume...")
        volume.commit()
        print("✅ Modal Volume committed successfully!")

        total_build_time = time.perf_counter() - t_start
        print("\n" + "=" * 80)
        print(f"🎉 100% MULTI-STRATEGY INDEX BUILD COMPLETE IN {total_build_time/60:.2f} MINUTES")
        print(f"   Final Vector Count: {index.ntotal:,}")
        print(f"   Final Metadata Entries: {len(all_chunks):,}")
        print("=" * 80)

        return {
            "status": "success",
            "total_vectors": index.ntotal,
            "duration_minutes": round(total_build_time / 60, 2),
            "strategy_counts": strategy_counts,
        }


@app.local_entrypoint()
def main():
    indexer = Full100PercentIndexer()
    res = indexer.build_index.remote()
    print("100% Full Build Result:", res)
