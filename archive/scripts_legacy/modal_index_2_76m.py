"""
Distributed 2.76M Passage Indexer on Modal Cloud
100% Hindi (1,000,000 passages) + 100% Marathi (1,000,000 passages) from ai4bharat/MSMARCO-XI
+ 766,666 English passages from microsoft/ms_marco v2.1

Target: ~2,766,666 Passages (~7.9GB FP16 index)
Passage-native chunking on A10G GPU, direct commit to Modal Volume voice-rag-index.

Run:
    .\\venv\\Scripts\\python.exe -m modal run scripts/modal_index_2_76m.py
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
        # Bake multilingual embedding model into container image
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-2-76m-indexer", image=image)


@app.cls(
    gpu="A10G",           # High-performance 24GB VRAM GPU
    memory=32768,         # 32GB system RAM
    timeout=14400,        # 4 hours max execution time
    volumes={"/index": volume},
)
class Cloud276MIndexer:

    @modal.method()
    def build_index(self, target_hi: int = 1_000_000, target_mr: int = 1_000_000, target_en: int = 766_666):
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

        total_target = target_hi + target_mr + target_en
        print("=" * 80)
        print(f"🚀 STARTING 2.76M PASSAGE INDEXING ON MODAL CLOUD")
        print(f"   Hindi Target:   {target_hi:,} passages (100%)")
        print(f"   Marathi Target: {target_mr:,} passages (100%)")
        print(f"   English Target: {target_en:,} passages")
        print(f"   Total Target:   {total_target:,} passages (~7.9GB FP16 index)")
        print("=" * 80)

        t_start = time.perf_counter()

        # ── 1. Load Multilingual Embedding Model onto GPU ──
        print("\nLoading multilingual-e5-base embedding model onto CUDA GPU (FP16)...")
        model = SentenceTransformer("/models/e5-base", device="cuda")
        model.max_seq_length = 64
        model.half()  # fp16 acceleration
        print("✅ Model loaded in FP16 on A10G GPU\n")

        all_texts = []
        all_metadata = []
        seen_hashes = set()

        # ── 2. Load Hindi & Marathi from MSMARCO-XI Direct Parquet ──
        lang_targets = [
            ("hi", "train/hintrain.parquet", "dev/hindev.parquet", target_hi),
            ("mr", "train/martrain.parquet", "dev/mardev.parquet", target_mr),
        ]

        for lang_code, train_file, dev_file, target_count in lang_targets:
            print(f"\n[{lang_code.upper()}] Loading up to {target_count:,} passages from ai4bharat/MSMARCO-XI...")
            count = 0
            t_lang_start = time.perf_counter()

            # Process train split, then dev split if needed to get 100%
            files_to_process = [train_file, dev_file]
            for filename in files_to_process:
                if count >= target_count:
                    break
                try:
                    print(f"  Downloading {filename}...")
                    t_dl = time.perf_counter()
                    pq_path = hf_hub_download(
                        repo_id="ai4bharat/MSMARCO-XI",
                        filename=filename,
                        repo_type="dataset"
                    )
                    print(f"  Downloaded in {time.perf_counter() - t_dl:.1f}s. Parsing Translated_passages...")

                    pf = pq.ParquetFile(pq_path)
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

                                if count >= target_count:
                                    break

                            if count >= target_count:
                                break

                        if count >= target_count:
                            break

                except Exception as e:
                    print(f"  ⚠️ Error processing {filename}: {e}")

            elapsed = time.perf_counter() - t_lang_start
            print(f"  ✅ [{lang_code.upper()}] Parsed {count:,} unique passages in {elapsed:.1f}s")

        # ── 3. Load English from microsoft/ms_marco v2.1 Parquet ──
        print(f"\n[EN] Downloading & parsing microsoft/ms_marco v2.1 (Target: {target_en:,})...")
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

                            if en_count >= target_en:
                                break

                        if en_count >= target_en:
                            break

                    if en_count >= target_en:
                        break

            except Exception as e:
                print(f"  ⚠️ Error processing {shard_name}: {e}")

            if en_count >= target_en:
                break

        elapsed = time.perf_counter() - t_en
        print(f"  ✅ [EN] Finished: {en_count:,} unique passages in {elapsed:.1f}s")

        total_passages = len(all_texts)
        print("\n" + "=" * 80)
        print(f"📦 TOTAL UNIQUE PASSAGES COLLECTED: {total_passages:,}")
        print("=" * 80)

        if total_passages == 0:
            raise RuntimeError("No passages were extracted. Check dataset sources.")

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
        print("🔍 RUNNING MULTI-LINGUAL SANITY VERIFICATION")
        print("=" * 80)

        test_queries = [
            ("What is the capital of France?", "en"),
            ("भारत की राजधानी क्या है?", "hi"),
            ("भारताची राजधानी कोणती आहे?", "mr"),
        ]

        for q_text, lang in test_queries:
            with torch.no_grad():
                qv = model.encode([q_text], normalize_embeddings=True)
            qv = np.asarray(qv, dtype=np.float32)
            scores, ids = index.search(qv, 3)
            print(f"\nQuery ({lang}): {q_text}")
            for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
                meta = all_metadata[idx]
                print(f"  Rank {rank+1} [score={score:.4f}, lang={meta['lang']}]: {meta['text'][:90]}...")

        total_time = time.perf_counter() - t_start
        print("\n" + "=" * 80)
        print(f"🎉 2.76M PASSAGE INDEX BUILD COMPLETE IN {total_time/60:.2f} MINUTES")
        print(f"   Final Vector Count: {index.ntotal:,}")
        print("=" * 80)
        return {
            "status": "success",
            "total_vectors": index.ntotal,
            "duration_minutes": round(total_time / 60, 2),
            "throughput_p_per_sec": round(throughput, 1),
        }


@app.local_entrypoint()
def main(target_hi: int = 1_000_000, target_mr: int = 1_000_000, target_en: int = 766_666):
    indexer = Cloud276MIndexer()
    res = indexer.build_index.remote(target_hi=target_hi, target_mr=target_mr, target_en=target_en)
    print("Build Result:", res)
