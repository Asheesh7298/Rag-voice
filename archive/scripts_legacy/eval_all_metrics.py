"""
Comprehensive evaluation script across all 13 Indic languages.
Measures:
  - Retrieval: Recall@1, Recall@5, Recall@10, MRR
  - QA: Exact Match (EM), Token F1
  - Guardrail: Decline rate, False Positive rate, Unsupported Answer rate

Run: python -m modal run scripts/eval_all_metrics.py
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
        "tqdm>=4.66.4",
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
app = modal.App("voice-rag-eval-metrics", image=image)


@app.cls(
    gpu="T4",
    volumes={"/index": volume},
    timeout=900,
)
class ComprehensiveEval:

    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import faiss

        device = 0 if torch.cuda.is_available() else -1

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index("/index/index.faiss")
        self.metadata = []
        with open("/index/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"✅ Index ready: {self.faiss_index.ntotal} vectors, {len(self.metadata)} metadata")

        qa_path = "/index/qa-model-finetuned" if os.path.exists("/index/qa-model-finetuned/model.safetensors") else "/models/qa-model"
        print(f"Loading QA model from {qa_path}...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device

        self.OFF_TOPIC_THRESHOLD  = 0.70
        self.MIN_RETRIEVAL_SCORE  = 0.65
        self.MIN_QA_SCORE         = 0.10
        self.MIN_ANSWER_RELEVANCE = 0.20
        self.TOP_K                = 10
        self.RERANK_TOP_N         = 50

        self.STOPWORDS = set(
            "का के की है में को और से हैं पर यह था थी थे "
            "इस कि एक भी ने जो वह हो तो कर इसके लिए अपने "
            "होता करने उनके साथ अगर अन्य कुछ तक जब "
            "the a an is are was were be been being have has had "
            "do does did will would shall should may might can could "
            "i me my we our you your he him his she her it its they them their "
            "what which who whom this that these those am "
            "in on at to for with from by of and or not no nor "
            "if but so than too very as how when where why all each every "
            "کا کی کے ہے میں "
            .split()
        )

    def _filter_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]

    def _hybrid_rerank(self, query: str, candidates: list):
        from rank_bm25 import BM25Okapi
        if not candidates: return []
        corpus = [self._filter_stopwords(c[0]["text"].split()) for c in candidates]
        query_tokens = self._filter_stopwords(query.split())
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(candidates)
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        combined = []
        for (meta, dense), bm25_s in zip(candidates, bm25_scores):
            combined.append({
                "text": meta["text"],
                "score": round(0.9 * dense + 0.1 * bm25_s / max_b, 4),
                "lang": meta["lang"],
                "strategy": meta["strategy"],
                "chunk_id": meta["chunk_id"],
                "query_id": meta["query_id"],
            })
        combined.sort(key=lambda c: c["score"], reverse=True)
        return combined

    def _retrieve(self, query: str):
        import numpy as np
        vec = self.embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        qv = vec.reshape(1, -1)
        scores, ids = self.faiss_index.search(qv, self.RERANK_TOP_N)
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            candidates.append((self.metadata[idx], float(score)))
        chunks = self._hybrid_rerank(query, candidates)[:self.TOP_K]
        return chunks, vec

    def _extract_best_answer(self, question: str, chunks: list) -> dict:
        import torch
        import torch.nn.functional as F
        if not chunks:
            return {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": ""}
        active_chunks = chunks[:5]
        questions = [question] * len(active_chunks)
        contexts = [c["text"] for c in active_chunks]
        inputs = self.qa_tokenizer(
            questions, contexts,
            return_tensors="pt", truncation=True, max_length=128, padding=True,
        )
        if self.qa_device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        s_logits = outputs.start_logits
        e_logits = outputs.end_logits
        best = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": ""}
        for i in range(len(active_chunks)):
            s = s_logits[i]
            e = e_logits[i]
            si = int(torch.argmax(s))
            ei = int(torch.argmax(e))
            if ei < si: ei = si
            toks = inputs["input_ids"][i][si:ei + 1]
            ans = self.qa_tokenizer.decode(toks, skip_special_tokens=True).strip()
            sp = float(F.softmax(s, dim=0)[si])
            ep = float(F.softmax(e, dim=0)[ei])
            sc = round(sp * ep, 4)
            if sc > best["score"] and ans:
                best = {"answer": ans, "score": sc, "chunk_idx": i, "source_text": active_chunks[i]["text"]}
        return best if best["score"] >= 0 else {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": ""}

    @modal.method()
    def evaluate(self, test_rows: list):
        import numpy as np
        from tqdm import tqdm

        print(f"\nRunning evaluation on {len(test_rows)} queries across 13 languages...")

        def normalize(t):
            return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()

        def compute_f1(pred, gold_list):
            if not gold_list: return 0.0, 0.0
            p_norm = normalize(pred)
            best_em = 0.0
            best_f1 = 0.0
            for g in gold_list:
                g_norm = normalize(g)
                if p_norm == g_norm and len(p_norm) > 0:
                    best_em = 1.0
                p_toks = p_norm.split()
                g_toks = g_norm.split()
                common = set(p_toks) & set(g_toks)
                if not p_toks or not g_toks:
                    f1 = 1.0 if p_toks == g_toks else 0.0
                else:
                    prec = len(common) / len(p_toks)
                    rec = len(common) / len(g_toks)
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                best_f1 = max(best_f1, f1)
            return best_em, best_f1

        results = []
        for row in tqdm(test_rows, desc="Evaluating"):
            q_text = row["query"]
            gold_qid = row["query_id"]
            gold_answers = row.get("answers", [])
            lang = row["lang"]

            chunks, qvec = self._retrieve(q_text)

            # Retrieval metrics
            retrieved_qids = [c["query_id"] for c in chunks]
            r1 = 1.0 if (retrieved_qids and retrieved_qids[0] == gold_qid) else 0.0
            r5 = 1.0 if gold_qid in retrieved_qids[:5] else 0.0
            r10 = 1.0 if gold_qid in retrieved_qids[:10] else 0.0

            mrr = 0.0
            for rank_idx, qid in enumerate(retrieved_qids):
                if qid == gold_qid:
                    mrr = 1.0 / (rank_idx + 1)
                    break

            # Guardrails
            top_score = chunks[0]["score"] if chunks else 0.0
            declined = False
            decline_reason = None

            if top_score < self.OFF_TOPIC_THRESHOLD:
                declined = True
                decline_reason = "off_topic"
            elif not chunks or top_score < self.MIN_RETRIEVAL_SCORE:
                declined = True
                decline_reason = "low_retrieval"

            pred_ans = ""
            qa_score = 0.0
            if not declined:
                best = self._extract_best_answer(q_text, chunks)
                qa_score = best["score"]
                if qa_score < self.MIN_QA_SCORE or not best["answer"]:
                    declined = True
                    decline_reason = "low_qa_confidence"
                else:
                    pred_ans = best["answer"]

            em, f1 = compute_f1(pred_ans, gold_answers) if not declined else (0.0, 0.0)

            # Check if answer was extracted from an unsupported / wrong passage
            unsupported = False
            false_positive = False
            if not declined and pred_ans:
                if gold_qid not in retrieved_qids[:5]:
                    unsupported = True
                    false_positive = True

            results.append({
                "query_id": gold_qid,
                "lang": lang,
                "r1": r1,
                "r5": r5,
                "r10": r10,
                "mrr": mrr,
                "em": em,
                "f1": f1,
                "declined": 1.0 if declined else 0.0,
                "decline_reason": decline_reason,
                "unsupported": 1.0 if unsupported else 0.0,
                "false_positive": 1.0 if false_positive else 0.0,
            })

        # Aggregate metrics
        from collections import defaultdict
        by_lang = defaultdict(list)
        for r in results:
            by_lang[r["lang"]].append(r)

        print("\n" + "=" * 90)
        print("COMPREHENSIVE MULTILINGUAL EVALUATION RESULTS")
        print("=" * 90)
        headers = f"{'Lang':<6} {'Queries':<8} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'EM':<8} {'F1':<8} {'Decl%':<8} {'Unsup%':<8}"
        print(headers)
        print("-" * 90)

        for lang in sorted(by_lang.keys()):
            l_rows = by_lang[lang]
            n = len(l_rows)
            r1 = np.mean([x["r1"] for x in l_rows]) * 100
            r5 = np.mean([x["r5"] for x in l_rows]) * 100
            r10 = np.mean([x["r10"] for x in l_rows]) * 100
            mrr = np.mean([x["mrr"] for x in l_rows]) * 100
            em = np.mean([x["em"] for x in l_rows]) * 100
            f1 = np.mean([x["f1"] for x in l_rows]) * 100
            decl = np.mean([x["declined"] for x in l_rows]) * 100
            unsup = np.mean([x["unsupported"] for x in l_rows]) * 100
            print(f"{lang:<6} {n:<8} {r1:>6.1f}% {r5:>6.1f}% {r10:>6.1f}% {mrr:>6.1f}% {em:>6.1f}% {f1:>6.1f}% {decl:>6.1f}% {unsup:>6.1f}%")

        print("-" * 90)
        tot_r1 = np.mean([x["r1"] for x in results]) * 100
        tot_r5 = np.mean([x["r5"] for x in results]) * 100
        tot_r10 = np.mean([x["r10"] for x in results]) * 100
        tot_mrr = np.mean([x["mrr"] for x in results]) * 100
        tot_em = np.mean([x["em"] for x in results]) * 100
        tot_f1 = np.mean([x["f1"] for x in results]) * 100
        tot_decl = np.mean([x["declined"] for x in results]) * 100
        tot_unsup = np.mean([x["unsupported"] for x in results]) * 100
        print(f"{'TOTAL':<6} {len(results):<8} {tot_r1:>6.1f}% {tot_r5:>6.1f}% {tot_r10:>6.1f}% {tot_mrr:>6.1f}% {tot_em:>6.1f}% {tot_f1:>6.1f}% {tot_decl:>6.1f}% {tot_unsup:>6.1f}%")
        print("=" * 90)


@app.local_entrypoint()
def main():
    import json, random
    rows = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]

    # Sample 30 queries per language (390 total queries)
    from collections import defaultdict
    by_lang = defaultdict(list)
    for r in rows:
        if r.get("query", "").strip() and r.get("answers"):
            by_lang[r["lang"]].append(r)

    random.seed(42)
    test_rows = []
    for lang, items in by_lang.items():
        random.shuffle(items)
        test_rows.extend(items[:30])

    print(f"Sampled {len(test_rows)} evaluation queries across {len(by_lang)} languages.")
    evaluator = ComprehensiveEval()
    evaluator.evaluate.remote(test_rows)
