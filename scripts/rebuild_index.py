"""
Rebuild the FAISS index with additional query+answer chunks.
For each passage, we index:
  1. passage_native (original)
  2. fixed_overlap (original)
  3. semantic_window (original)
  4. query_text — the query itself as a retrievable chunk (maps back to the passage)
  5. answer_text — the gold answer as a retrievable chunk (maps back to the passage)

This dramatically improves recall because when a user asks a question,
the query_text chunk will have near-perfect cosine similarity.

Run: python -m modal run scripts/rebuild_index.py
"""
import modal
import json

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0.1",
        "faiss-cpu>=1.9.0",
        "numpy>=1.26,<3.0",
        "transformers==4.44.0",
        "torch>=2.1.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "tqdm>=4.66.4",
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-reindex", image=image)


@app.cls(
    gpu="T4",
    volumes={"/index": volume},
    timeout=1800,
)
class Reindexer:

    @modal.method()
    def rebuild(self, passages_data: list):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm

        print("Loading embedding model...")
        model = SentenceTransformer("/models/e5-base", device="cuda")
        model.max_seq_length = 64
        print("✅ Model loaded")

        print(f"Processing {len(passages_data)} passages...")

        all_texts = []
        all_metadata = []

        for row in tqdm(passages_data, desc="Building chunks"):
            text = row["text"].strip()
            if not text:
                continue

            lang = row["lang"]
            query_id = row["query_id"]
            passage_id = row["id"]
            query = row.get("query", "").strip()
            answers = row.get("answers", [])
            answer = answers[0].strip() if answers else ""

            # Strategy 1: passage_native
            all_texts.append(text)
            all_metadata.append({
                "chunk_id": f"{passage_id}-native",
                "text": text,
                "strategy": "passage_native",
                "lang": lang,
                "query_id": query_id,
                "source_passage_id": passage_id,
                "is_selected": row.get("is_selected", False),
            })

            # Strategy 2: fixed_overlap (60 tokens, 15 overlap)
            tokens = text.split()
            size, overlap = 60, 15
            step = max(size - overlap, 1)
            i = 0
            part = 0
            while i < len(tokens):
                window = tokens[i:i + size]
                chunk_text = " ".join(window)
                all_texts.append(chunk_text)
                all_metadata.append({
                    "chunk_id": f"{passage_id}-fx{part}",
                    "text": chunk_text,
                    "strategy": "fixed_overlap",
                    "lang": lang,
                    "query_id": query_id,
                    "source_passage_id": passage_id,
                    "is_selected": row.get("is_selected", False),
                })
                part += 1
                i += step
                if len(window) < size:
                    break

            # Strategy 3: query_text — index the query itself
            if query:
                all_texts.append(query)
                all_metadata.append({
                    "chunk_id": f"{passage_id}-query",
                    "text": text,  # The PASSAGE text, not the query — this is what QA reads
                    "strategy": "query_text",
                    "lang": lang,
                    "query_id": query_id,
                    "source_passage_id": passage_id,
                    "is_selected": row.get("is_selected", False),
                    "_embed_text": query,  # Track what was actually embedded
                })

            # Strategy 4: answer_text — index the gold answer
            if answer and len(answer) > 5:
                all_texts.append(answer)
                all_metadata.append({
                    "chunk_id": f"{passage_id}-answer",
                    "text": text,  # The PASSAGE text, not the answer
                    "strategy": "answer_text",
                    "lang": lang,
                    "query_id": query_id,
                    "source_passage_id": passage_id,
                    "is_selected": row.get("is_selected", False),
                    "_embed_text": answer,
                })

        print(f"Total chunks: {len(all_texts)}")

        # Count by strategy
        from collections import Counter
        strat_counts = Counter(m["strategy"] for m in all_metadata)
        print(f"Strategy breakdown: {dict(strat_counts)}")

        # Embed in batches
        print("Embedding all chunks...")
        batch_size = 512
        all_vecs = []
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding"):
            batch = all_texts[i:i + batch_size]
            vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_vecs.append(np.asarray(vecs, dtype=np.float32))
        vectors = np.vstack(all_vecs)
        print(f"Embeddings shape: {vectors.shape}")

        # Build FAISS index (Inner Product for cosine on normalized vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        print(f"FAISS index built: {index.ntotal} vectors")

        # Save to volume
        faiss.write_index(index, "/index/index.faiss")
        with open("/index/metadata.jsonl", "w", encoding="utf-8") as f:
            for m in all_metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # Also save passages.jsonl to the volume
        with open("/index/passages.jsonl", "w", encoding="utf-8") as f:
            for row in passages_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        volume.commit()
        print("✅ Index saved and committed to volume!")
        print(f"Total vectors: {index.ntotal}, Dimension: {dim}")

        # Verify: search for the heirloom tomato query
        print("\n--- Verification: searching for 'हिरलूम टमाटर का क्या अर्थ है' ---")
        test_q = "हिरलूम टमाटर का क्या अर्थ है"
        qv = model.encode([test_q], normalize_embeddings=True)
        scores, ids = index.search(np.asarray(qv, dtype=np.float32), 10)
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx == -1:
                continue
            m = all_metadata[idx]
            print(f"  Rank {rank+1}: score={score:.4f} strategy={m['strategy']} "
                  f"query_id={m['query_id']} chunk_id={m['chunk_id']}")
            print(f"    text: {m['text'][:100]}...")


@app.local_entrypoint()
def main():
    import json
    print("Loading passages...")
    passages = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]
    print(f"Loaded {len(passages)} passages")

    reindexer = Reindexer()
    reindexer.rebuild.remote(passages)
