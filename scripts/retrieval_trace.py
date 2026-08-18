"""
Comprehensive Retrieval Tracing & Multi-Configuration Evaluation.

For every query, logs:
  1. Query text
  2. Expected/gold answer
  3. Top-10 retrieved chunk IDs, languages, dense/BM25/hybrid scores
  4. Final top-5 chunks passed to QA
  5. Gold passage presence at top-1/5/10/20/50
  6. QA selected answer & confidence
  7. Failure classification:
     - RETRIEVAL_FAILURE: Gold passage not in dense top-50
     - RERANK_FAILURE: Gold in dense top-50 but pushed out of final top-k by reranking
     - QA_FAILURE: Gold in final QA context but wrong answer extracted
     - DATASET_FAILURE: Expected answer/context does not exist
     - CONFIDENCE_FAILURE: Wrong answer returned with high confidence (score_diff > 0)

Tests retrieval configurations:
  A. Current E5 dense + BM25 hybrid (0.85/0.15)
  B. E5 dense-only (no BM25)
  C. E5 + BM25 with Indic stopword filtering (already active) + language-match boost
  D. E5 top-50 → language-filtered rerank → top-5
  E. E5-small (intfloat/multilingual-e5-small)

Run: python -m modal run scripts/retrieval_trace.py
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
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base'); "
        "SentenceTransformer('intfloat/multilingual-e5-small').save('/models/e5-small')"
        "\"",
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; "
        "AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model'); "
        "AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-retrieval-trace", image=image)


# ── Stopwords for BM25 filtering ──────────────────────────────────────────────
STOPWORDS = set(
    # Hindi
    "का के की है में को और से हैं पर यह था थी थे "
    "इस कि एक भी ने जो वह हो तो कर इसके लिए अपने "
    "होता करने उनके साथ अगर अन्य कुछ तक जब "
    # Marathi
    "आहे आणि व या ची चे चा च्या साठी तर मग "
    "नाही पण म्हणून जर तर तो ती ते त्या काय कसे "
    # English
    "the a an is are was were be been being have has had "
    "do does did will would shall should may might can could "
    "i me my we our you your he him his she her it its they them their "
    "what which who whom this that these those am "
    "in on at to for with from by of and or not no nor "
    "if but so than too very as how when where why all each every "
    # Urdu
    "کا کی کے ہے میں "
    .split()
)


def detect_lang_family(text: str) -> list[str]:
    """Detect the Indic language family from Unicode script ranges."""
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


def normalize_text(t: str) -> str:
    return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()


def compute_f1(pred: str, gold_list: list[str]) -> tuple[float, float]:
    if not gold_list:
        return 0.0, 0.0
    p_norm = normalize_text(pred)
    best_em = 0.0
    best_f1 = 0.0
    for g in gold_list:
        g_norm = normalize_text(g)
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


def filter_stopwords(tokens: list) -> list:
    return [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]


@app.cls(
    gpu="T4",
    volumes={"/index": volume},
    timeout=3600,
)
class RetrievalTracer:

    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import faiss
        import numpy as np

        device = 0 if torch.cuda.is_available() else -1
        self.device_str = "cuda" if device == 0 else "cpu"
        print(f"Device: {self.device_str}")

        # ── E5-base (primary) ──
        print("Loading E5-base...")
        self.embed_model = SentenceTransformer("/models/e5-base", device=self.device_str)
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)

        # ── E5-small (config E) ──
        print("Loading E5-small...")
        self.embed_small = SentenceTransformer("/models/e5-small", device=self.device_str)
        self.embed_small.max_seq_length = 64
        self.embed_small.encode(["warmup"], normalize_embeddings=True)

        # ── FAISS index & metadata ──
        print("Loading FAISS index & metadata...")
        self.faiss_index = faiss.read_index("/index/index.faiss")
        self.metadata = []
        with open("/index/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"Index ready: {self.faiss_index.ntotal} vectors, {len(self.metadata)} metadata rows")

        # ── Pre-compute E5-small FAISS index from metadata texts ──
        print("Building E5-small index from metadata texts...")
        all_texts = [m["text"] for m in self.metadata]
        batch_size = 512
        small_vecs = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            vecs = self.embed_small.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            small_vecs.append(np.asarray(vecs, dtype=np.float32))
        vectors = np.vstack(small_vecs)
        dim = vectors.shape[1]
        self.faiss_small = faiss.IndexFlatIP(dim)
        self.faiss_small.add(vectors)
        print(f"E5-small index built: {self.faiss_small.ntotal} vectors, dim={dim}")

        # ── QA model ──
        qa_path = "/index/qa-model-finetuned" if os.path.exists("/index/qa-model-finetuned/model.safetensors") else "/models/qa-model"
        print(f"Loading QA model from {qa_path}...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device
        print("✅ All models loaded")

    # ── FAISS search ──────────────────────────────────────────────────────────

    def _search_faiss(self, query: str, top_n: int, use_small: bool = False):
        import numpy as np
        model = self.embed_small if use_small else self.embed_model
        index = self.faiss_small if use_small else self.faiss_index
        vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        qv = vec.reshape(1, -1)
        scores, ids = index.search(qv, top_n)
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            candidates.append((self.metadata[idx], float(score)))
        return candidates

    # ── BM25 reranking ────────────────────────────────────────────────────────

    def _hybrid_rerank(self, query: str, candidates: list,
                       lang_boost_langs: list = None,
                       use_stopwords: bool = True,
                       dense_weight: float = 0.85,
                       bm25_weight: float = 0.15,
                       lang_boost_val: float = 0.05):
        from rank_bm25 import BM25Okapi
        if not candidates:
            return []

        if use_stopwords:
            corpus = [filter_stopwords(c[0]["text"].split()) for c in candidates]
            query_tokens = filter_stopwords(query.split())
        else:
            corpus = [c[0]["text"].split() for c in candidates]
            query_tokens = query.split()

        if query_tokens and any(corpus):
            bm25 = BM25Okapi(corpus)
            bm25_scores = bm25.get_scores(query_tokens)
        else:
            bm25_scores = [0.0] * len(candidates)

        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        combined = []
        for (meta, dense), bm25_s in zip(candidates, bm25_scores):
            score = dense_weight * dense + bm25_weight * (bm25_s / max_b)
            if lang_boost_langs and meta.get("lang") in lang_boost_langs:
                score += lang_boost_val
            combined.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "dense_score": round(dense, 4),
                "bm25_score": round(bm25_s, 4),
                "bm25_norm": round(bm25_s / max_b, 4),
                "hybrid_score": round(score, 4),
                "lang": meta["lang"],
                "strategy": meta["strategy"],
                "query_id": meta["query_id"],
            })
        combined.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return combined

    def _dense_only_rank(self, candidates: list):
        """Config B: Pure dense ranking, no BM25."""
        combined = []
        for meta, dense in candidates:
            combined.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "dense_score": round(dense, 4),
                "bm25_score": 0.0,
                "bm25_norm": 0.0,
                "hybrid_score": round(dense, 4),
                "lang": meta["lang"],
                "strategy": meta["strategy"],
                "query_id": meta["query_id"],
            })
        combined.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return combined

    # ── QA extraction ─────────────────────────────────────────────────────────

    def _extract_qa(self, question: str, chunks: list) -> dict:
        import torch
        import torch.nn.functional as F
        if not chunks:
            return {"answer": "", "score": 0.0, "score_diff": -99.0, "null_score": 0.0, "chunk_idx": -1, "source_lang": None}

        active = chunks[:5]
        best = {"answer": "", "score": -1.0, "score_diff": -99.0, "null_score": 0.0, "chunk_idx": -1, "source_lang": None}

        for i, chunk in enumerate(active):
            inputs = self.qa_tokenizer(
                question, chunk["text"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            if self.qa_device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            s = outputs.start_logits[0]
            e = outputs.end_logits[0]

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
            toks = inputs["input_ids"][0][best_si:best_ei + 1]
            ans = self.qa_tokenizer.decode(toks, skip_special_tokens=True).strip()

            sp = float(F.softmax(s, dim=0)[best_si])
            ep = float(F.softmax(e, dim=0)[best_ei])
            confidence = round(sp * ep, 4)

            if ans and score_diff > best["score_diff"]:
                best = {
                    "answer": ans,
                    "score": confidence,
                    "score_diff": round(score_diff, 4),
                    "null_score": round(null_score, 4),
                    "chunk_idx": i,
                    "source_lang": chunk["lang"],
                }

        return best

    # ── Failure classification ────────────────────────────────────────────────

    def _classify_failure(self, gold_qid: str, gold_answers: list,
                          dense_top50_qids: list,
                          reranked_top5_qids: list,
                          qa_result: dict,
                          f1_threshold: float = 0.4) -> tuple[str, float, float]:
        """Returns (failure_category, em, f1)."""
        pred = qa_result["answer"]
        em, f1 = compute_f1(pred, gold_answers)
        is_correct = f1 >= f1_threshold

        if is_correct:
            return "CORRECT", em, f1

        # Check dense retrieval
        gold_in_dense_50 = gold_qid in dense_top50_qids
        gold_in_rerank_5 = gold_qid in reranked_top5_qids

        if not gold_in_dense_50:
            return "RETRIEVAL_FAILURE", em, f1
        elif not gold_in_rerank_5:
            return "RERANK_FAILURE", em, f1
        else:
            # Gold is in final QA context but wrong answer
            if qa_result["score_diff"] > 0 and qa_result["score"] > 0.3:
                return "CONFIDENCE_FAILURE", em, f1
            else:
                return "QA_FAILURE", em, f1

    # ── Per-query trace ───────────────────────────────────────────────────────

    def _trace_single_query(self, query: str, gold_qid: str, gold_answers: list,
                            query_lang: str, config: str) -> dict:
        """Run one query through a specific config and return full trace."""
        lang_fam = detect_lang_family(query)

        if config == "A":
            # Current: E5-base dense + BM25 hybrid (0.85/0.15) with stopword filter
            candidates = self._search_faiss(query, top_n=50)
            reranked = self._hybrid_rerank(query, candidates, lang_boost_langs=lang_fam,
                                           use_stopwords=True)
        elif config == "B":
            # Dense-only, no BM25
            candidates = self._search_faiss(query, top_n=50)
            reranked = self._dense_only_rank(candidates)
        elif config == "C":
            # E5 + BM25 with stopwords + stronger language boost (0.10)
            candidates = self._search_faiss(query, top_n=50)
            reranked = self._hybrid_rerank(query, candidates, lang_boost_langs=lang_fam,
                                           use_stopwords=True, lang_boost_val=0.10)
        elif config == "D":
            # E5 top-50 → hard language filter → rerank → top-5
            candidates = self._search_faiss(query, top_n=50)
            # Hard filter to query language family
            lang_filtered = [(m, s) for m, s in candidates if m.get("lang") in lang_fam]
            if not lang_filtered:
                lang_filtered = candidates  # fallback
            reranked = self._hybrid_rerank(query, lang_filtered, lang_boost_langs=lang_fam,
                                           use_stopwords=True)
        elif config == "E":
            # E5-small
            candidates = self._search_faiss(query, top_n=50, use_small=True)
            reranked = self._hybrid_rerank(query, candidates, lang_boost_langs=lang_fam,
                                           use_stopwords=True)
        else:
            raise ValueError(f"Unknown config: {config}")

        # Dense top-50 query_ids (before reranking)
        dense_top50_qids = [c[0]["query_id"] for c in candidates]

        # Reranked query_ids
        reranked_qids = [c["query_id"] for c in reranked]

        # Gold passage position tracking
        gold_dense_rank = None
        for rank_idx, qid in enumerate(dense_top50_qids):
            if qid == gold_qid:
                gold_dense_rank = rank_idx + 1
                break

        gold_rerank_rank = None
        for rank_idx, qid in enumerate(reranked_qids):
            if qid == gold_qid:
                gold_rerank_rank = rank_idx + 1
                break

        # Top-10 retrieved chunks detail
        top10_detail = []
        for rank_idx, chunk in enumerate(reranked[:10]):
            top10_detail.append({
                "rank": rank_idx + 1,
                "chunk_id": chunk["chunk_id"],
                "lang": chunk["lang"],
                "strategy": chunk["strategy"],
                "dense_score": chunk["dense_score"],
                "bm25_score": chunk["bm25_score"],
                "bm25_norm": chunk["bm25_norm"],
                "hybrid_score": chunk["hybrid_score"],
                "query_id": chunk["query_id"],
                "text_preview": chunk["text"][:120],
            })

        # QA on top-5
        qa_chunks = reranked[:5]
        qa_result = self._extract_qa(query, qa_chunks)

        # Language consistency: what % of top-10 match query language?
        top10_langs = [c["lang"] for c in reranked[:10]]
        lang_match_count = sum(1 for l in top10_langs if l in lang_fam)
        lang_consistency = round(lang_match_count / max(1, len(top10_langs)), 2)

        # Classify failure
        cat, em, f1 = self._classify_failure(
            gold_qid, gold_answers,
            dense_top50_qids[:50],
            reranked_qids[:5],
            qa_result
        )

        # Recall at various k
        r1 = gold_qid in dense_top50_qids[:1]
        r5 = gold_qid in dense_top50_qids[:5]
        r10 = gold_qid in dense_top50_qids[:10]
        r20 = gold_qid in dense_top50_qids[:20]
        r50 = gold_qid in dense_top50_qids[:50]

        # MRR
        mrr = 0.0
        for ri, qid in enumerate(dense_top50_qids):
            if qid == gold_qid:
                mrr = 1.0 / (ri + 1)
                break

        return {
            "query": query,
            "query_lang": query_lang,
            "detected_family": lang_fam,
            "gold_qid": gold_qid,
            "gold_answers": [a[:80] for a in gold_answers],
            "config": config,
            # Retrieval metrics
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "recall_at_20": r20,
            "recall_at_50": r50,
            "mrr": round(mrr, 4),
            "gold_dense_rank": gold_dense_rank,
            "gold_rerank_rank": gold_rerank_rank,
            # Top-10 chunks
            "top10_chunks": top10_detail,
            "top10_languages": top10_langs,
            "lang_consistency": lang_consistency,
            # QA results
            "qa_answer": qa_result["answer"],
            "qa_confidence": qa_result["score"],
            "qa_score_diff": qa_result["score_diff"],
            "qa_null_score": qa_result["null_score"],
            "qa_source_lang": qa_result["source_lang"],
            "qa_chunk_idx": qa_result["chunk_idx"],
            # Accuracy
            "em": em,
            "f1": round(f1, 4),
            "failure_category": cat,
        }

    # ── Main evaluation methods ───────────────────────────────────────────────

    @modal.method()
    def trace_specific_queries(self, test_cases: list, configs: list[str]):
        """Detailed trace for specific failure queries across all configs."""
        results = []
        for tc in test_cases:
            for cfg in configs:
                trace = self._trace_single_query(
                    tc["query"], tc.get("query_id", ""),
                    tc.get("answers", []),
                    tc.get("lang", ""), cfg
                )
                results.append(trace)
        return results

    @modal.method()
    def evaluate_all_configs(self, test_rows: list, configs: list[str]):
        """Run full evaluation across sampled queries for each config."""
        from collections import defaultdict
        from tqdm import tqdm
        import numpy as np

        all_config_results = {}

        for cfg in configs:
            print(f"\n{'='*60}")
            print(f"EVALUATING CONFIG {cfg}")
            print(f"{'='*60}")

            per_lang = defaultdict(lambda: {
                "n": 0, "r1": 0, "r5": 0, "r10": 0, "r20": 0, "r50": 0,
                "mrr": [], "em": 0, "f1": [], "correct": 0,
                "lang_consistency": [],
                "failures": defaultdict(int),
            })

            failure_traces = []  # Collect detailed traces for failures

            for row in tqdm(test_rows, desc=f"Config {cfg}"):
                trace = self._trace_single_query(
                    row["query"], row["query_id"],
                    row.get("answers", []),
                    row["lang"], cfg
                )

                lang = row["lang"]
                stats = per_lang[lang]
                stats["n"] += 1
                stats["r1"] += int(trace["recall_at_1"])
                stats["r5"] += int(trace["recall_at_5"])
                stats["r10"] += int(trace["recall_at_10"])
                stats["r20"] += int(trace["recall_at_20"])
                stats["r50"] += int(trace["recall_at_50"])
                stats["mrr"].append(trace["mrr"])
                stats["em"] += int(trace["em"])
                stats["f1"].append(trace["f1"])
                stats["correct"] += int(trace["failure_category"] == "CORRECT")
                stats["lang_consistency"].append(trace["lang_consistency"])
                stats["failures"][trace["failure_category"]] += 1

                # Collect detailed failure traces (not CORRECT)
                if trace["failure_category"] != "CORRECT":
                    failure_traces.append({
                        "query": trace["query"],
                        "lang": trace["query_lang"],
                        "gold_qid": trace["gold_qid"],
                        "gold_answers": trace["gold_answers"],
                        "category": trace["failure_category"],
                        "qa_answer": trace["qa_answer"][:80],
                        "qa_confidence": trace["qa_confidence"],
                        "qa_score_diff": trace["qa_score_diff"],
                        "gold_dense_rank": trace["gold_dense_rank"],
                        "gold_rerank_rank": trace["gold_rerank_rank"],
                        "top5_langs": trace["top10_languages"][:5],
                        "lang_consistency": trace["lang_consistency"],
                    })

            # Aggregate per-config results
            summary_rows = []
            total_n = 0
            total_r1 = total_r5 = total_r10 = total_r20 = total_r50 = 0
            total_em = total_correct = 0
            all_mrr = []
            all_f1 = []
            all_lang_con = []
            total_failures = defaultdict(int)

            for lang in sorted(per_lang.keys()):
                s = per_lang[lang]
                n = s["n"]
                total_n += n
                total_r1 += s["r1"]
                total_r5 += s["r5"]
                total_r10 += s["r10"]
                total_r20 += s["r20"]
                total_r50 += s["r50"]
                total_em += s["em"]
                total_correct += s["correct"]
                all_mrr.extend(s["mrr"])
                all_f1.extend(s["f1"])
                all_lang_con.extend(s["lang_consistency"])
                for fcat, cnt in s["failures"].items():
                    total_failures[fcat] += cnt

                summary_rows.append({
                    "lang": lang,
                    "n": n,
                    "r1": round(s["r1"] / n * 100, 1),
                    "r5": round(s["r5"] / n * 100, 1),
                    "r10": round(s["r10"] / n * 100, 1),
                    "r20": round(s["r20"] / n * 100, 1),
                    "r50": round(s["r50"] / n * 100, 1),
                    "mrr": round(float(np.mean(s["mrr"])) * 100, 1),
                    "em": round(s["em"] / n * 100, 1),
                    "f1": round(float(np.mean(s["f1"])) * 100, 1),
                    "acc": round(s["correct"] / n * 100, 1),
                    "lang_con": round(float(np.mean(s["lang_consistency"])) * 100, 1),
                    "failures": dict(s["failures"]),
                })

            total_summary = {
                "lang": "TOTAL",
                "n": total_n,
                "r1": round(total_r1 / total_n * 100, 1),
                "r5": round(total_r5 / total_n * 100, 1),
                "r10": round(total_r10 / total_n * 100, 1),
                "r20": round(total_r20 / total_n * 100, 1),
                "r50": round(total_r50 / total_n * 100, 1),
                "mrr": round(float(np.mean(all_mrr)) * 100, 1),
                "em": round(total_em / total_n * 100, 1),
                "f1": round(float(np.mean(all_f1)) * 100, 1),
                "acc": round(total_correct / total_n * 100, 1),
                "lang_con": round(float(np.mean(all_lang_con)) * 100, 1),
                "failures": dict(total_failures),
            }

            all_config_results[cfg] = {
                "by_lang": summary_rows,
                "total": total_summary,
                "failure_traces": failure_traces[:50],  # Cap at 50 detailed traces
            }

        return all_config_results


@app.local_entrypoint()
def main():
    import json, random

    # ── Phase 1: Detailed trace on the 7 reported failure queries ─────────

    # We need to find gold query_ids for known queries in the dataset
    passages = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]
    query_to_passage = {}
    for p in passages:
        q = p.get("query", "").strip()
        if q:
            query_to_passage[q] = p

    # The 7 reported failures — these are OUT-OF-DATASET queries (Category C)
    # but we trace them anyway to see what gets retrieved
    ood_cases = [
        {"query": "भारत की राजधानी क्या है?", "lang": "hi", "query_id": "", "answers": ["नई दिल्ली"]},
        {"query": "भारताची राजधानी कोणती आहे?", "lang": "mr", "query_id": "", "answers": ["नवी दिल्ली"]},
        {"query": "what is photosynthesis?", "lang": "en", "query_id": "", "answers": ["the process by which green plants use sunlight to synthesize foods"]},
        {"query": "what are the main parts of an atom?", "lang": "en", "query_id": "", "answers": ["protons, neutrons and electrons"]},
        {"query": "what are symptoms of diabetes?", "lang": "en", "query_id": "", "answers": ["increased thirst, frequent urination, hunger, fatigue"]},
        {"query": "what is the cost of tile installation per square foot?", "lang": "en", "query_id": "", "answers": ["$2 to $14"]},
        {"query": "what is a normal blood pressure reading?", "lang": "en", "query_id": "", "answers": ["120/80 mmHg"]},
    ]

    runner = RetrievalTracer()

    print("\n" + "=" * 100)
    print("PHASE 1: DETAILED TRACE ON 7 REPORTED FAILURE QUERIES (all configs)")
    print("=" * 100)

    ood_traces = runner.trace_specific_queries.remote(ood_cases, ["A", "B", "C", "D"])
    for t in ood_traces:
        print(f"\n{'─'*80}")
        print(f"Config {t['config']} | Query [{t['query_lang']}]: {t['query']}")
        print(f"  Gold answers: {t['gold_answers']}")
        print(f"  Detected family: {t['detected_family']}")
        print(f"  Category: {t['failure_category']}")
        print(f"  QA answer: {t['qa_answer']!r} (conf={t['qa_confidence']}, score_diff={t['qa_score_diff']}, src_lang={t['qa_source_lang']})")
        print(f"  Lang consistency (top-10): {t['lang_consistency']*100:.0f}%")
        print(f"  Top-5 languages: {t['top10_languages'][:5]}")
        for ch in t["top10_chunks"][:5]:
            print(f"    Rank {ch['rank']} [{ch['lang']}] dense={ch['dense_score']} bm25_n={ch['bm25_norm']} hybrid={ch['hybrid_score']} strat={ch['strategy']} text: {ch['text_preview'][:80]}")

    # ── Phase 2: Full 13-language benchmark across configs A-E ────────────

    from collections import defaultdict
    by_lang = defaultdict(list)
    for p in passages:
        if p.get("query", "").strip() and p.get("answers"):
            by_lang[p["lang"]].append(p)

    random.seed(42)
    sample_rows = []
    for lang, items in by_lang.items():
        random.shuffle(items)
        sample_rows.extend(items[:30])  # 30 per language × 13 = 390

    print(f"\n\n{'='*100}")
    print(f"PHASE 2: FULL 13-LANGUAGE BENCHMARK ({len(sample_rows)} queries × 5 configs)")
    print(f"{'='*100}")

    config_results = runner.evaluate_all_configs.remote(sample_rows, ["A", "B", "C", "D", "E"])

    # Print results for each config
    for cfg in ["A", "B", "C", "D", "E"]:
        res = config_results[cfg]
        cfg_names = {
            "A": "E5-base + BM25 hybrid (current)",
            "B": "E5-base dense-only (no BM25)",
            "C": "E5-base + BM25 + strong lang boost (0.10)",
            "D": "E5-base + hard lang filter + rerank",
            "E": "E5-small + BM25 hybrid",
        }
        print(f"\n{'='*100}")
        print(f"CONFIG {cfg}: {cfg_names[cfg]}")
        print(f"{'='*100}")
        hdr = f"{'Lang':<6}{'N':<5}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@20':>7}{'R@50':>7}{'MRR':>7}{'EM':>7}{'F1':>7}{'Acc':>7}{'LangCon':>9}"
        print(hdr)
        print("-" * 100)
        for row in res["by_lang"]:
            print(
                f"{row['lang']:<6}{row['n']:<5}"
                f"{row['r1']:>6.1f}%{row['r5']:>6.1f}%{row['r10']:>6.1f}%"
                f"{row['r20']:>6.1f}%{row['r50']:>6.1f}%{row['mrr']:>6.1f}%"
                f"{row['em']:>6.1f}%{row['f1']:>6.1f}%{row['acc']:>6.1f}%"
                f"{row['lang_con']:>8.1f}%"
            )
        print("-" * 100)
        t = res["total"]
        print(
            f"{t['lang']:<6}{t['n']:<5}"
            f"{t['r1']:>6.1f}%{t['r5']:>6.1f}%{t['r10']:>6.1f}%"
            f"{t['r20']:>6.1f}%{t['r50']:>6.1f}%{t['mrr']:>6.1f}%"
            f"{t['em']:>6.1f}%{t['f1']:>6.1f}%{t['acc']:>6.1f}%"
            f"{t['lang_con']:>8.1f}%"
        )
        print(f"\nFailure distribution:")
        for fcat, cnt in sorted(t["failures"].items()):
            print(f"  {fcat}: {cnt} ({cnt/t['n']*100:.1f}%)")

    # ── Phase 3: Cross-config comparison summary ──────────────────────────

    print(f"\n\n{'='*100}")
    print("CROSS-CONFIG COMPARISON SUMMARY")
    print(f"{'='*100}")
    hdr = f"{'Config':<8}{'Description':<45}{'R@1':>7}{'R@5':>7}{'MRR':>7}{'F1':>7}{'Acc':>7}{'LangCon':>9}"
    print(hdr)
    print("-" * 100)
    cfg_names = {
        "A": "E5-base + BM25 hybrid (current)",
        "B": "E5-base dense-only",
        "C": "E5-base + BM25 + strong lang boost",
        "D": "E5-base + hard lang filter + rerank",
        "E": "E5-small + BM25 hybrid",
    }
    for cfg in ["A", "B", "C", "D", "E"]:
        t = config_results[cfg]["total"]
        print(
            f"{cfg:<8}{cfg_names[cfg]:<45}"
            f"{t['r1']:>6.1f}%{t['r5']:>6.1f}%{t['mrr']:>6.1f}%"
            f"{t['f1']:>6.1f}%{t['acc']:>6.1f}%{t['lang_con']:>8.1f}%"
        )

    # ── Phase 4: Sample failure traces ────────────────────────────────────

    print(f"\n\n{'='*100}")
    print("SAMPLE FAILURE TRACES (Config A, first 20)")
    print(f"{'='*100}")
    traces_a = config_results["A"].get("failure_traces", [])
    for i, ft in enumerate(traces_a[:20]):
        print(f"\n{i+1}. [{ft['lang']}] {ft['query']}")
        print(f"   Gold: {ft['gold_answers']}")
        print(f"   Got:  {ft['qa_answer']!r} (conf={ft['qa_confidence']}, score_diff={ft['qa_score_diff']})")
        print(f"   Category: {ft['category']}")
        print(f"   Gold dense rank: {ft['gold_dense_rank']}, Gold rerank rank: {ft['gold_rerank_rank']}")
        print(f"   Top-5 langs: {ft['top5_langs']}, Lang consistency: {ft['lang_consistency']*100:.0f}%")
