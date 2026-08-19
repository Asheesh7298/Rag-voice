"""
Distributed 3.0M Passage Indexer on Modal Cloud (Option A: 100% Balanced Multi-Lingual)
- 100% Hindi (~1,000,000 passages from ai4bharat/MSMARCO-XI)
- 100% Marathi (~1,000,000 passages from ai4bharat/MSMARCO-XI)
- 1,000,000 English passages (from microsoft/ms_marco v2.1)

Total: ~3.0 Million Passages
Estimated Runtime: ~20-22 min on A10G GPU (~$0.38 compute cost)
"""

import modal
import os

# ── Modal Container Image ───────────────────────────────────────────────────
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
        # Bake multilingual embedding model into the container image
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-3m-indexer", image=image)


@app.cls(
    gpu="A10G",           # 24GB VRAM GPU with FP16 Tensor Cores
    memory=32768,         # 32GB RAM buffer for 3.0M float32 vectors (~8.6 GB)
    timeout=10800,        # 3 hours maximum execution time
    volumes={"/index": volume},
)
class Cloud3MIndexer:

    @modal.method()
    def build_index(self, en_target: int = 1_000_000):
        import time
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
        print("🚀 STARTING 100% BALANCED 3.0M PASSAGE INDEXING ON MODAL CLOUD")
        print("=" * 80)
        print("Dataset targets:")
        print("  • Hindi:   100% of all available data in MSMARCO-XI")
        print("  • Marathi: 100% of all available data in MSMARCO-XI")
        print(f"  • English: {en_target:,} passages from MS MARCO v2.1\n")

        t_start = time.perf_counter()

        # ── 1. Load Multilingual Embedding Model onto GPU ──
        print("Loading multilingual embedding model onto A10G GPU (FP16)...")
        model = SentenceTransformer("/models/e5-base", device="cuda")
        model.max_seq_length = 64
        model.half()  # fp16 for 2x faster matrix multiplication
        print("✅ Model loaded in FP16 on A10G GPU\n")

        all_texts = []
        all_metadata = []
        seen_hashes = set()

        # ── 2. Stream 100% Hindi & 100% Marathi from MSMARCO-XI ──
        lang_files = [
            ("hi", "train/hintrain.parquet"),
            ("mr", "train/martrain.parquet"),
        ]

        for lang_code, filename in lang_files:
            print(f"[{lang_code.upper()}] Downloading full {filename} from ai4bharat/MSMARCO-XI...")
            t_dl = time.perf_counter()
            try:
                pq_path = hf_hub_download(
                    repo_id="ai4bharat/MSMARCO-XI",
                    filename=filename,
                    repo_type="dataset"
                )
                print(f"  Downloaded in {time.perf_counter() - t_dl:.1f}s. Parsing 100% of rows with PyArrow...")

                pf = pq.ParquetFile(pq_path)
                count = 0
                t_parse = time.perf_counter()

                for batch in pf.iter_batches(batch_size=10000, columns=["query_id", "passages", "query", "Answer"]):
                    df_dict = batch.to_pydict()
                    qids = df_dict.get("query_id", [])
                    passages_list = df_dict.get("passages", [])
                    queries = df_dict.get("query", [])
                    answers_list = df_dict.get("Answer", [])

                    for idx in range(len(passages_list)):
                        p_struct = passages_list[idx]
                        if not isinstance(p_struct, dict):
                            continue
                        p_texts = p_struct.get("Translated_passages", [])
                        is_sel_list = p_struct.get("is_selected", [])

                        qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(count)
                        q_text = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                        ans = answers_list[idx] if idx < len(answers_list) and answers_list[idx] else ""
                        ans_text = str(ans).strip() if ans else ""

                        for p_idx, p_text in enumerate(p_texts):
                            text = str(p_text).strip()
                            if not text or len(text) < 20:
                                continue

                            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                            if text_hash in seen_hashes:
                                continue
                            seen_hashes.add(text_hash)

                            is_sel = bool(is_sel_list[p_idx]) if is_sel_list and p_idx < len(is_sel_list) else False

                            all_texts.append(text)
                            all_metadata.append({
                                "chunk_id": f"{lang_code}-{qid}-p{p_idx}",
                                "text": text,
                                "lang": lang_code,
                                "query_id": f"{lang_code}-{qid}",
                                "query": q_text,
                                "answer": ans_text,
                                "is_selected": is_sel,
                            })
                            count += 1

                elapsed = time.perf_counter() - t_parse
                print(f"  ✅ [{lang_code.upper()}] Parsed 100% ({count:,} unique passages) in {elapsed:.1f}s")

            except Exception as e:
                print(f"  ❌ Error processing {lang_code}: {e}")

        # ── 3. Load English from microsoft/ms_marco v2.1 Parquet ──
        print(f"\n[EN] Downloading & parsing microsoft/ms_marco v2.1 (Target: {en_target:,})...")
        t_en = time.perf_counter()
        en_count = 0

        # Files v2.1/train-00000 to train-00006
        for shard_idx in range(7):
            shard_name = f"v2.1/train-{shard_idx:05d}-of-00007.parquet"
            try:
                print(f"  Downloading {shard_name}...")
                shard_path = hf_hub_download(
                    repo_id="microsoft/ms_marco",
                    filename=shard_name,
                    repo_type="dataset"
                )
                pf = pq.ParquetFile(shard_path)

                for batch in pf.iter_batches(batch_size=10000, columns=["query_id", "passages", "query", "answers"]):
                    df_dict = batch.to_pydict()
                    qids = df_dict.get("query_id", [])
                    passages_list = df_dict.get("passages", [])
                    queries = df_dict.get("query", [])
                    answers_list = df_dict.get("answers", [])

                    for idx in range(len(passages_list)):
                        p_struct = passages_list[idx]
                        if not isinstance(p_struct, dict):
                            continue
                        p_texts = p_struct.get("passage_text", [])
                        is_sel_list = p_struct.get("is_selected", [])

                        qid = str(qids[idx]) if idx < len(qids) and qids[idx] is not None else str(en_count)
                        q_text = str(queries[idx]).strip() if idx < len(queries) and queries[idx] else ""
                        ans = answers_list[idx] if idx < len(answers_list) and answers_list[idx] else []
                        ans_text = str(ans[0]).strip() if isinstance(ans, list) and ans else ""
                        if ans_text.lower() == "no answer present.":
                            ans_text = ""

                        for p_idx, p_text in enumerate(p_texts):
                            text = str(p_text).strip()
                            if not text or len(text) < 20:
                                continue

                            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                            if text_hash in seen_hashes:
                                continue
                            seen_hashes.add(text_hash)

                            is_sel = bool(is_sel_list[p_idx]) if is_sel_list and p_idx < len(is_sel_list) else False

                            all_texts.append(text)
                            all_metadata.append({
                                "chunk_id": f"en-{qid}-p{p_idx}",
                                "text": text,
                                "lang": "en",
                                "query_id": f"en-{qid}",
                                "query": q_text,
                                "answer": ans_text,
                                "is_selected": is_sel,
                            })
                            en_count += 1

                            if en_count >= en_target:
                                break

                        if en_count >= en_target:
                            break

                    if en_count >= en_target:
                        break

            except Exception as e:
                print(f"  ⚠️ Error processing {shard_name}: {e}")

            if en_count >= en_target:
                break

        elapsed = time.perf_counter() - t_en
        print(f"  ✅ [EN] Finished: {en_count:,} unique passages in {elapsed:.1f}s")

        total_passages = len(all_texts)
        print("\n" + "=" * 80)
        print(f"📦 TOTAL UNIQUE PASSAGES COLLECTED (100% BALANCED): {total_passages:,}")
        print("=" * 80)

        # ── 4. High-Throughput GPU Batch Embedding ──
        batch_size = 512
        all_vecs = []
        t_embed = time.perf_counter()
        print(f"\n⚡ Encoding {total_passages:,} passages on A10G GPU (Batch size: {batch_size})...")

        for i in tqdm(range(0, total_passages, batch_size), desc="GPU Embedding"):
            batch = all_texts[i:i + batch_size]
            with torch.no_grad():
                vecs = model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
            all_vecs.append(np.asarray(vecs, dtype=np.float32))

        vectors = np.vstack(all_vecs)
        t_embed_end = time.perf_counter()
        embed_duration = t_embed_end - t_embed
        throughput = total_passages / max(embed_duration, 1)
        print(f"\n✅ Embedding completed in {embed_duration/60:.2f} min ({throughput:.1f} passages/sec)")
        print(f"Matrix shape: {vectors.shape} ({vectors.nbytes / (1024**3):.2f} GB)")

        # ── 5. Build FAISS Index ──
        dim = vectors.shape[1]
        print(f"\n⚙️ Building FAISS IndexFlatIP (dim={dim})...")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        print(f"✅ FAISS index ready with {index.ntotal:,} vectors")

        # ── 6. Save directly to Modal Volume ──
        print("\n💾 Writing to Modal Volume '/index'...")
        os.makedirs("/index", exist_ok=True)

        faiss.write_index(index, "/index/index.faiss")
        with open("/index/metadata.jsonl", "w", encoding="utf-8") as f:
            for m in all_metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print("Committing changes to Modal Volume...")
        volume.commit()
        print("✅ Modal Volume committed successfully!")

        # ── 7. Multilingual Sanity Verification ──
        print("\n" + "=" * 80)
        print("🔍 RUNNING MULTI-LINGUAL VERIFICATION SEARCH")
        print("=" * 80)

        test_queries = [
            ("What is the capital of France?", "en"),
            ("भारत का प्रधानमंत्री कौन है?", "hi"),
            ("महाराष्ट्राची राजधानी कोणती आहे?", "mr"),
        ]

        for query_text, lang in test_queries:
            q_vec = model.encode([query_text], normalize_embeddings=True)
            q_vec = np.asarray(q_vec, dtype=np.float32)
            scores, indices = index.search(q_vec, 3)

            print(f"\nQuery ({lang}): '{query_text}'")
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(all_metadata):
                    continue
                meta = all_metadata[idx]
                print(f"  Rank {rank+1} [Score: {score:.4f}] [{meta['lang']}] {meta['text'][:90]}...")

        total_elapsed = time.perf_counter() - t_start
        print("\n" + "=" * 80)
        print(f"🎉 100% BALANCED 3.0M PASSAGE INDEXING COMPLETE IN {total_elapsed/60:.2f} MINUTES!")
        print("=" * 80)


@app.local_entrypoint()
def main(en_target: int = 1_000_000):
    indexer = Cloud3MIndexer()
    indexer.build_index.remote(en_target=en_target)
