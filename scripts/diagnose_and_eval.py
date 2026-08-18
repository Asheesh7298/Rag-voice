"""
Diagnostic and Comprehensive Evaluation across all 13 Indic Languages.
Classifies failures into:
  - Category A: Retrieval failure (Gold passage not in top-k)
  - Category B: QA failure (Gold passage in top-k, but QA selects wrong span)
  - Category C: Dataset failure (Query/context does not exist in dataset)
  - Category D: Confidence/Guardrail failure (System gives high confidence on irrelevant/unsupported passage)

Measures:
  - Retrieval: Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, MRR
  - Answer: Exact Match (EM), Token F1, Accuracy, Unsupported Answer Rate, False Positive Rate
  - Per-language breakdown across all 13 Indic languages
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
app = modal.App("voice-rag-diagnose-eval", image=image)


@app.cls(
    gpu="T4",
    volumes={"/index": volume},
    timeout=1800,
)
class FullPipelineDiagnostic:

    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import faiss

        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading embedding model on {'cuda' if device == 0 else 'cpu'}...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)

        print("Loading FAISS index & metadata...")
        self.faiss_index = faiss.read_index("/index/index.faiss")
        self.metadata = []
        with open("/index/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"Index ready: {self.faiss_index.ntotal} vectors, {len(self.metadata)} metadata rows")

        qa_path = "/index/qa-model-finetuned" if os.path.exists("/index/qa-model-finetuned/model.safetensors") else "/models/qa-model"
        print(f"Loading QA model from {qa_path}...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device

    def _detect_lang_family(self, text: str) -> list[str]:
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                return ["hi", "mr", "ne"]
            elif 0x0980 <= cp <= 0x09FF:
                return ["bn", "as"]
            elif 0x0A00 <= cp <= 0x0A7F:
                return ["pa"]
            elif 0x0A80 <= cp <= 0x0AFF:
                return ["gu"]
            elif 0x0B00 <= cp <= 0x0B7F:
                return ["or"]
            elif 0x0B80 <= cp <= 0x0BFF:
                return ["ta"]
            elif 0x0C00 <= cp <= 0x0C7F:
                return ["te"]
            elif 0x0C80 <= cp <= 0x0CFF:
                return ["kn"]
            elif 0x0D00 <= cp <= 0x0D7F:
                return ["ml"]
            elif 0x0600 <= cp <= 0x06FF:
                return ["ur"]
        return ["en"]

    def _search_faiss(self, query: str, top_n: int = 50):
        import numpy as np
        vec = self.embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        qv = vec.reshape(1, -1)
        scores, ids = self.faiss_index.search(qv, top_n)
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            candidates.append((self.metadata[idx], float(score)))
        return candidates

    def _hybrid_rerank(self, query: str, candidates: list, lang_boost: list = None):
        from rank_bm25 import BM25Okapi
        if not candidates: return []
        corpus = [c[0]["text"].split() for c in candidates]
        query_tokens = query.split()
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(candidates)
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        combined = []
        for (meta, dense), bm25_s in zip(candidates, bm25_scores):
            score = 0.85 * dense + 0.15 * (bm25_s / max_b)
            if lang_boost and meta.get("lang") in lang_boost:
                score += 0.05
            combined.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "dense_score": round(dense, 4),
                "bm25_score": round(bm25_s, 4),
                "score": round(score, 4),
                "lang": meta["lang"],
                "strategy": meta["strategy"],
                "query_id": meta["query_id"],
            })
        combined.sort(key=lambda c: c["score"], reverse=True)
        return combined

    def _extract_qa(self, question: str, chunks: list):
        import torch
        import torch.nn.functional as F
        if not chunks:
            return {"answer": "", "score": 0.0, "null_score": 0.0, "score_diff": -99.0, "chunk_idx": 0, "source_text": "", "lang": None}

        active = chunks[:5]
        questions = [question] * len(active)
        contexts = [c["text"] for c in active]
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

        best = {"answer": "", "score": -1.0, "null_score": 0.0, "score_diff": -99.0, "chunk_idx": 0, "source_text": "", "lang": None}

        for i in range(len(active)):
            s = s_logits[i]
            e = e_logits[i]

            # SQuAD 2.0 null (no-answer) logit at token 0 (<s>)
            null_score = float(s[0] + e[0])

            seq_len = s.shape[0]
            best_span_score = -1e9
            best_si, best_ei = 0, 0

            for si in range(1, seq_len):
                for ei in range(si, min(si + 30, seq_len)):
                    span_score = float(s[si] + e[ei])
                    if span_score > best_span_score:
                        best_span_score = span_score
                        best_si, best_ei = si, ei

            score_diff = best_span_score - null_score

            toks = inputs["input_ids"][i][best_si:best_ei + 1]
            ans = self.qa_tokenizer.decode(toks, skip_special_tokens=True).strip()

            sp = float(F.softmax(s, dim=0)[best_si])
            ep = float(F.softmax(e, dim=0)[best_ei])
            confidence = round(sp * ep, 4)

            if best_span_score > -1e8 and ans:
                if score_diff > best["score_diff"]:
                    best = {
                        "answer": ans,
                        "score": confidence,
                        "null_score": round(null_score, 4),
                        "score_diff": round(score_diff, 4),
                        "chunk_idx": i,
                        "source_text": active[i]["text"],
                        "lang": active[i]["lang"],
                    }

        return best

    @modal.method()
    def inspect_queries(self, test_cases: list):
        """Diagnose specific queries in detail."""
        reports = []
        for tc in test_cases:
            query = tc["query"]
            gold_qid = tc.get("query_id")
            gold_answers = tc.get("answers", [])

            lang_fam = self._detect_lang_family(query)
            candidates_50 = self._search_faiss(query, top_n=50)

            ranks = [c[0]["query_id"] for c in candidates_50]
            in_top1 = (gold_qid in ranks[:1]) if gold_qid else False
            in_top5 = (gold_qid in ranks[:5]) if gold_qid else False
            in_top10 = (gold_qid in ranks[:10]) if gold_qid else False
            in_top20 = (gold_qid in ranks[:20]) if gold_qid else False
            in_top50 = (gold_qid in ranks[:50]) if gold_qid else False

            reranked_10 = self._hybrid_rerank(query, candidates_50, lang_boost=lang_fam)[:10]
            qa_res = self._extract_qa(query, reranked_10)

            # Failure classification
            failure_cat = None
            if not gold_qid or not gold_answers:
                failure_cat = "Category C: Dataset failure (Query/context not in dataset)"
                if qa_res["score"] > 0.5:
                    failure_cat += " + Category D: Confidence/guardrail failure (Confidently wrong answer returned)"
            elif not in_top10:
                failure_cat = "Category A: Retrieval failure (Gold passage not in top-10)"
            else:
                ans_norm = "".join(c.lower() for c in qa_res["answer"] if c.isalnum() or c.isspace()).strip()
                match = any(ans_norm in "".join(c.lower() for c in g if c.isalnum() or c.isspace()) for g in gold_answers)
                if not match:
                    failure_cat = "Category B: QA failure (Gold passage in top-10, but QA selected wrong span)"
                else:
                    failure_cat = "Success: Correct answer extracted"

            top5_chunks = [
                {
                    "rank": r_idx + 1,
                    "lang": c["lang"],
                    "dense_score": c["dense_score"],
                    "bm25_score": c["bm25_score"],
                    "hybrid_score": c["score"],
                    "text": c["text"][:120],
                }
                for r_idx, c in enumerate(reranked_10[:5])
            ]

            reports.append({
                "query": query,
                "gold_qid": gold_qid,
                "gold_answers": gold_answers,
                "lang_detected": lang_fam,
                "top5_chunks": top5_chunks,
                "gold_in_top1": in_top1,
                "gold_in_top5": in_top5,
                "gold_in_top10": in_top10,
                "gold_in_top20": in_top20,
                "gold_in_top50": in_top50,
                "qa_span": qa_res["answer"],
                "qa_confidence": qa_res["score"],
                "qa_score_diff": qa_res["score_diff"],
                "qa_source_lang": qa_res["lang"],
                "failure_category": failure_cat,
            })
        return reports

    @modal.method()
    def full_evaluation(self, test_rows: list):
        """Run evaluation across all 13 Indic languages."""
        import numpy as np
        from collections import defaultdict
        from tqdm import tqdm

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
        failure_counts = defaultdict(int)

        for row in tqdm(test_rows, desc="Evaluating"):
            q_text = row["query"]
            gold_qid = row["query_id"]
            gold_answers = row.get("answers", [])
            lang = row["lang"]

            lang_fam = self._detect_lang_family(q_text)
            candidates_50 = self._search_faiss(q_text, top_n=50)

            c_qids = [c[0]["query_id"] for c in candidates_50]
            r1 = 1.0 if (c_qids and c_qids[0] == gold_qid) else 0.0
            r5 = 1.0 if gold_qid in c_qids[:5] else 0.0
            r10 = 1.0 if gold_qid in c_qids[:10] else 0.0
            r20 = 1.0 if gold_qid in c_qids[:20] else 0.0
            r50 = 1.0 if gold_qid in c_qids[:50] else 0.0

            mrr = 0.0
            for rank_idx, qid in enumerate(c_qids):
                if qid == gold_qid:
                    mrr = 1.0 / (rank_idx + 1)
                    break

            reranked_10 = self._hybrid_rerank(q_text, candidates_50, lang_boost=lang_fam)[:10]
            qa_res = self._extract_qa(q_text, reranked_10)

            pred_ans = qa_res["answer"]
            em, f1 = compute_f1(pred_ans, gold_answers)

            is_correct = (f1 >= 0.4)
            cat = "Correct"
            if not is_correct:
                if gold_qid not in c_qids[:10]:
                    cat = "Category A: Retrieval failure"
                elif qa_res["score_diff"] < 0:
                    cat = "Category D: Unanswerable rejected"
                else:
                    cat = "Category B: QA span extraction failure"
            failure_counts[cat] += 1

            unsupported = (not is_correct and pred_ans and gold_qid not in c_qids[:5])
            false_pos = (pred_ans and not is_correct)

            results.append({
                "lang": lang,
                "r1": r1, "r5": r5, "r10": r10, "r20": r20, "r50": r50, "mrr": mrr,
                "em": em, "f1": f1, "accuracy": 1.0 if is_correct else 0.0,
                "unsupported": 1.0 if unsupported else 0.0,
                "false_positive": 1.0 if false_pos else 0.0,
                "category": cat,
            })

        by_lang = defaultdict(list)
        for r in results:
            by_lang[r["lang"]].append(r)

        summary_table = []
        for lang in sorted(by_lang.keys()):
            l_rows = by_lang[lang]
            n = len(l_rows)
            summary_table.append({
                "lang": lang,
                "n": n,
                "r1": float(np.mean([x["r1"] for x in l_rows]) * 100),
                "r5": float(np.mean([x["r5"] for x in l_rows]) * 100),
                "r10": float(np.mean([x["r10"] for x in l_rows]) * 100),
                "r20": float(np.mean([x["r20"] for x in l_rows]) * 100),
                "r50": float(np.mean([x["r50"] for x in l_rows]) * 100),
                "mrr": float(np.mean([x["mrr"] for x in l_rows]) * 100),
                "em": float(np.mean([x["em"] for x in l_rows]) * 100),
                "f1": float(np.mean([x["f1"] for x in l_rows]) * 100),
                "acc": float(np.mean([x["accuracy"] for x in l_rows]) * 100),
                "unsup": float(np.mean([x["unsupported"] for x in l_rows]) * 100),
                "fp": float(np.mean([x["false_positive"] for x in l_rows]) * 100),
            })

        total_row = {
            "lang": "TOTAL",
            "n": len(results),
            "r1": float(np.mean([x["r1"] for x in results]) * 100),
            "r5": float(np.mean([x["r5"] for x in results]) * 100),
            "r10": float(np.mean([x["r10"] for x in results]) * 100),
            "r20": float(np.mean([x["r20"] for x in results]) * 100),
            "r50": float(np.mean([x["r50"] for x in results]) * 100),
            "mrr": float(np.mean([x["mrr"] for x in results]) * 100),
            "em": float(np.mean([x["em"] for x in results]) * 100),
            "f1": float(np.mean([x["f1"] for x in results]) * 100),
            "acc": float(np.mean([x["accuracy"] for x in results]) * 100),
            "unsup": float(np.mean([x["unsupported"] for x in results]) * 100),
            "fp": float(np.mean([x["false_positive"] for x in results]) * 100),
        }

        return {
            "by_lang": summary_table,
            "total": total_row,
            "failure_distribution": dict(failure_counts),
        }


@app.local_entrypoint()
def main():
    import json, random

    # Specific test cases requested by user
    failed_cases = [
        {"query": "भारत की राजधानी क्या है?", "lang": "hi"},
        {"query": "भारताची राजधानी कोणती आहे?", "lang": "mr"},
        {"query": "what is photosynthesis?", "lang": "en"},
        {"query": "what are the main parts of an atom?", "lang": "en"},
        {"query": "what are symptoms of diabetes?", "lang": "en"},
        {"query": "what is the cost of tile installation per square foot?", "lang": "en"},
        {"query": "what is a normal blood pressure reading?", "lang": "en"},
    ]

    print("Running deep diagnostic on the 7 reported failure queries...")
    runner = FullPipelineDiagnostic()
    diagnostics = runner.inspect_queries.remote(failed_cases)

    print("\n" + "="*80)
    print("DETAILED FAILURE INSPECTION REPORT (7 QUERIES)")
    print("="*80)
    for rep in diagnostics:
        print(f"\nQuery: {rep['query']}")
        print(f"Classification: {rep['failure_category']}")
        print(f"Detected Script Family: {rep['lang_detected']}")
        print(f"QA Span Extracted: {rep['qa_span']!r} (confidence: {rep['qa_confidence']}, score_diff: {rep['qa_score_diff']}, source_lang: {rep['qa_source_lang']})")
        print("Top 5 Retrieved Chunks:")
        for ch in rep["top5_chunks"]:
            print(f"  Rank {ch['rank']} [{ch['lang']}] dense={ch['dense_score']} bm25={ch['bm25_score']} hybrid={ch['hybrid_score']} text: {ch['text']}")

    # Run full 13-language benchmark
    passages = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]
    from collections import defaultdict
    by_lang = defaultdict(list)
    for p in passages:
        if p.get("query", "").strip() and p.get("answers"):
            by_lang[p["lang"]].append(p)

    random.seed(42)
    sample_rows = []
    for lang, items in by_lang.items():
        random.shuffle(items)
        sample_rows.extend(items[:30])

    print(f"\nRunning 13-language benchmark on {len(sample_rows)} sampled queries...")
    metrics = runner.full_evaluation.remote(sample_rows)

    print("\n" + "="*100)
    print("MULTILINGUAL 13-LANGUAGE EVALUATION METRICS")
    print("="*100)
    header = f"{'Lang':<6} {'N':<5} {'R@1':<7} {'R@5':<7} {'R@10':<7} {'R@20':<7} {'R@50':<7} {'MRR':<7} {'EM':<7} {'F1':<7} {'Acc':<7} {'Unsup':<7} {'FP':<7}"
    print(header)
    print("-"*100)
    for row in metrics["by_lang"]:
        print(f"{row['lang']:<6} {row['n']:<5} {row['r1']:>5.1f}% {row['r5']:>5.1f}% {row['r10']:>5.1f}% {row['r20']:>5.1f}% {row['r50']:>5.1f}% {row['mrr']:>5.1f}% {row['em']:>5.1f}% {row['f1']:>5.1f}% {row['acc']:>5.1f}% {row['unsup']:>5.1f}% {row['fp']:>5.1f}%")
    print("-"*100)
    t = metrics["total"]
    print(f"{t['lang']:<6} {t['n']:<5} {t['r1']:>5.1f}% {t['r5']:>5.1f}% {t['r10']:>5.1f}% {t['r20']:>5.1f}% {t['r50']:>5.1f}% {t['mrr']:>5.1f}% {t['em']:>5.1f}% {t['f1']:>5.1f}% {t['acc']:>5.1f}% {t['unsup']:>5.1f}% {t['fp']:>5.1f}%")
    print("="*100)
    print(f"\nFailure Distribution across {t['n']} evaluation queries:")
    for k, v in metrics["failure_distribution"].items():
        print(f"  {k}: {v} ({v/t['n']*100:.1f}%)")
