"""
Deep diagnostic of the RAG pipeline for specific queries.
Runs inside Modal to access the actual index and models.
"""
import modal
import json
import os

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
        "rank-bm25>=0.2.2",
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\"",
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; "
        "AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model'); "
        "AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-debug", image=image)

@app.cls(
    gpu="T4",
    volumes={"/index": volume},
    timeout=600,
)
class DebugRAG:

    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import faiss
        import numpy as np

        device = 0 if torch.cuda.is_available() else -1

        # Embedding model
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)

        # FAISS index
        self.faiss_index = faiss.read_index("/index/index.faiss")
        self.metadata = []
        with open("/index/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"Index: {self.faiss_index.ntotal} vectors, {len(self.metadata)} metadata entries")
        print(f"Index metric: {self.faiss_index.metric_type}")

        # QA model
        qa_path = "/index/qa-model-finetuned" if os.path.exists("/index/qa-model-finetuned/model.safetensors") else "/models/qa-model"
        print(f"QA model: {qa_path}")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device

    @modal.method()
    def diagnose(self, query: str, expected_query_id: str = None):
        import torch
        import torch.nn.functional as F
        import numpy as np
        from rank_bm25 import BM25Okapi

        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"EXPECTED QUERY_ID: {expected_query_id}")
        print(f"{'='*80}")

        # Step 1: Embed the query
        print("\n--- STEP 1: QUERY EMBEDDING ---")
        qvec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        qvec = qvec.astype(np.float32)
        norm = np.linalg.norm(qvec)
        print(f"Embedding norm: {norm:.6f}")
        print(f"Embedding shape: {qvec.shape}")
        print(f"First 5 dims: {qvec[:5]}")

        # Step 2: FAISS search - top 20
        print("\n--- STEP 2: FAISS SEARCH (top 20) ---")
        qv = qvec.reshape(1, -1)
        scores, ids = self.faiss_index.search(qv, 20)

        faiss_results = []
        found_expected = False
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            is_match = expected_query_id and meta.get("query_id") == expected_query_id
            if is_match:
                found_expected = True
            faiss_results.append((meta, float(score), rank))
            marker = " *** EXPECTED ***" if is_match else ""
            print(f"  Rank {rank+1}: score={score:.4f} lang={meta.get('lang','?')} "
                  f"query_id={meta.get('query_id','?')} "
                  f"chunk_id={meta.get('chunk_id','?')} "
                  f"strategy={meta.get('strategy','?')}{marker}")
            print(f"    text: {meta['text'][:120]}...")

        if not found_expected and expected_query_id:
            print(f"\n  ⚠️  Expected query_id '{expected_query_id}' NOT in top 20!")
            # Search through ALL metadata
            print("  Searching entire index for expected query_id...")
            for i, m in enumerate(self.metadata):
                if m.get("query_id") == expected_query_id:
                    # Get the embedding distance
                    chunk_vec = np.zeros((1, qvec.shape[0]), dtype=np.float32)
                    # We can't easily get the stored vector, but we can re-embed
                    chunk_emb = self.embed_model.encode([m["text"]], normalize_embeddings=True)[0]
                    chunk_emb = chunk_emb.astype(np.float32)
                    cos_sim = float(np.dot(qvec, chunk_emb))
                    print(f"  Found at index {i}: cos_sim={cos_sim:.4f} lang={m['lang']} chunk_id={m.get('chunk_id','?')}")
                    print(f"    text: {m['text'][:200]}...")
                    print(f"    strategy: {m.get('strategy','?')}")

        # Step 3: BM25 reranking
        print("\n--- STEP 3: BM25 HYBRID RERANKING ---")
        candidates = [(self.metadata[idx], float(score)) for score, idx in zip(scores[0], ids[0]) if idx != -1]
        corpus = [c[0]["text"].split() for c in candidates]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query.split())
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        combined = []
        for rank, ((meta, dense), bm25_s) in enumerate(zip(candidates, bm25_scores)):
            hybrid = round(0.7 * dense + 0.3 * bm25_s / max_b, 4)
            combined.append({
                "text": meta["text"],
                "score": hybrid,
                "dense_score": dense,
                "bm25_score": round(bm25_s / max_b, 4),
                "lang": meta.get("lang"),
                "strategy": meta.get("strategy"),
                "chunk_id": meta.get("chunk_id"),
                "query_id": meta.get("query_id"),
                "faiss_rank": rank,
            })
        combined.sort(key=lambda c: c["score"], reverse=True)

        for rank, c in enumerate(combined):
            is_match = expected_query_id and c.get("query_id") == expected_query_id
            marker = " *** EXPECTED ***" if is_match else ""
            print(f"  Rank {rank+1}: hybrid={c['score']:.4f} (dense={c['dense_score']:.4f}, bm25={c['bm25_score']:.4f}) "
                  f"lang={c['lang']} query_id={c['query_id']} "
                  f"faiss_rank={c['faiss_rank']+1}{marker}")

        # Step 4: QA extraction on top 8
        print("\n--- STEP 4: EXTRACTIVE QA ON TOP 8 ---")
        top_k = combined[:8]
        for rank, chunk in enumerate(top_k[:5]):
            inputs = self.qa_tokenizer(
                query, chunk["text"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            if self.qa_device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            start_idx = int(torch.argmax(start_logits))
            end_idx = int(torch.argmax(end_logits))
            if end_idx < start_idx:
                end_idx = start_idx

            tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()

            start_prob = float(F.softmax(start_logits, dim=0)[start_idx])
            end_prob = float(F.softmax(end_logits, dim=0)[end_idx])
            score = round(start_prob * end_prob, 4)

            is_match = expected_query_id and chunk.get("query_id") == expected_query_id
            marker = " *** EXPECTED ***" if is_match else ""
            print(f"  Chunk {rank+1} (query_id={chunk['query_id']}, hybrid={chunk['score']:.4f}):{marker}")
            print(f"    start_idx={start_idx}, end_idx={end_idx}")
            print(f"    start_prob={start_prob:.4f}, end_prob={end_prob:.4f}, joint_score={score:.4f}")
            print(f"    answer: '{answer}'")
            print(f"    chunk_text: '{chunk['text'][:150]}...'")
            print()

        # Step 5: Guardrail check
        print("\n--- STEP 5: GUARDRAIL ANALYSIS ---")
        top_score = combined[0]["score"] if combined else 0.0
        OFF_TOPIC_THRESHOLD = 0.25
        MIN_RETRIEVAL_SCORE = 0.20
        MIN_QA_SCORE = 0.15
        print(f"Top hybrid score: {top_score:.4f}")
        print(f"OFF_TOPIC_THRESHOLD: {OFF_TOPIC_THRESHOLD} -> {'PASS' if top_score >= OFF_TOPIC_THRESHOLD else 'DECLINE (off_topic)'}")
        print(f"MIN_RETRIEVAL_SCORE: {MIN_RETRIEVAL_SCORE} -> {'PASS' if top_score >= MIN_RETRIEVAL_SCORE else 'DECLINE (low_retrieval)'}")

        # Find the best QA answer
        best_score = 0.0
        best_answer = ""
        for rank, chunk in enumerate(top_k[:5]):
            inputs = self.qa_tokenizer(
                query, chunk["text"],
                return_tensors="pt", truncation=True, max_length=128, padding=True,
            )
            if self.qa_device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            s = outputs.start_logits[0]
            e = outputs.end_logits[0]
            si = int(torch.argmax(s))
            ei = int(torch.argmax(e))
            if ei < si:
                ei = si
            tokens = inputs["input_ids"][0][si:ei + 1]
            ans = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()
            sp = float(F.softmax(s, dim=0)[si])
            ep = float(F.softmax(e, dim=0)[ei])
            sc = round(sp * ep, 4)
            if sc > best_score and ans:
                best_score = sc
                best_answer = ans

        print(f"Best QA score: {best_score:.4f}")
        print(f"MIN_QA_SCORE: {MIN_QA_SCORE} -> {'PASS' if best_score >= MIN_QA_SCORE else 'DECLINE (low_qa_confidence)'}")
        print(f"Best answer: '{best_answer}'")

        # Step 6: Check metadata structure
        print("\n--- STEP 6: METADATA STRUCTURE SAMPLE ---")
        sample = self.metadata[0]
        print(f"Metadata keys: {list(sample.keys())}")
        print(f"First entry: lang={sample.get('lang')}, query_id={sample.get('query_id')}, "
              f"chunk_id={sample.get('chunk_id')}, strategy={sample.get('strategy')}, "
              f"text_len={len(sample.get('text',''))}")

        # Check how many chunks share the same query_id
        print("\n--- STEP 7: CHUNKS PER QUERY_ID ---")
        from collections import Counter
        qid_counts = Counter(m.get("query_id") for m in self.metadata)
        count_dist = Counter(qid_counts.values())
        print(f"Total unique query_ids: {len(qid_counts)}")
        print(f"Distribution of chunks per query_id: {dict(sorted(count_dist.items()))}")
        if expected_query_id:
            print(f"Chunks for {expected_query_id}: {qid_counts.get(expected_query_id, 0)}")

@app.local_entrypoint()
def main():
    rag = DebugRAG()

    # Test 1: Hindi heirloom tomato query
    rag.diagnose.remote("हिरलूम टमाटर का क्या अर्थ है", expected_query_id="hi-10440")

    # Test 2: English prime minister query (not in dataset)
    rag.diagnose.remote("who is the prime minister of india", expected_query_id=None)
