import modal
import json
import time
import os
import sys
from collections import defaultdict

# ── Image definition matching modal_app.py exactly ──
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.112.0",
        "uvicorn[standard]>=0.30.5",
        "pydantic>=2.8.2",
        "python-dotenv>=1.0.1",
        "httpx>=0.27.0",
        "tenacity>=8.5.0",
        "rank-bm25>=0.2.2",
        "python-multipart>=0.0.9",
        "tqdm>=4.66.4",
        "sentence-transformers>=3.0.1",
        "faiss-cpu>=1.9.0",
        "numpy>=1.26,<3.0",
        "transformers==4.44.0",
        "torch>=2.1.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    )
    .run_commands(
        # Bake embedding model into image at build time
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')"
        "\"",
        # Bake extractive QA model into image at build time
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; "
        "AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model'); "
        "AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
secrets = [modal.Secret.from_name("voice-rag-secrets")]

app = modal.App("voice-rag", image=image, secrets=secrets)

# Mount local src directory to import strategies.py in cloud
eval_image = (
    image
    .add_local_file("modal_app.py", "/root/modal_app.py")
    .add_local_dir("src", "/root/src")
)

@app.cls(
    gpu="T4",
    image=eval_image,
    volumes={"/index": volume},
    timeout=900,  # Increase class-level timeout to 15 minutes
)
class EvalVoiceRAG:

    @modal.enter()
    def load(self):
        import os, torch
        from sentence_transformers import SentenceTransformer
        from transformers import (
            AutoTokenizer, AutoModelForQuestionAnswering
        )
        import faiss, json

        device = 0 if torch.cuda.is_available() else -1
        print(f"Device: {'cuda' if device == 0 else 'cpu'}")

        # Config from Modal secrets
        self.OFF_TOPIC_THRESHOLD      = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.70"))
        self.MIN_RETRIEVAL_SCORE      = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.65"))
        self.MIN_QA_SCORE             = float(os.getenv("MIN_QA_SCORE", "0.10"))
        self.TOP_K                    = int(os.getenv("TOP_K", "10"))
        self.RERANK_TOP_N             = int(os.getenv("RERANK_TOP_N", "50"))
        self.SARVAM_KEY               = os.getenv("SARVAM_API_KEY", "")
        self.SARVAM_URL               = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")

        # Indic + English stopwords for BM25 filtering
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
        
        # Default fallback config
        self.index_dir = "/index"
        self.use_prefixes = False

        # ── Embedding model ──
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)
        print("✅ Embedding model ready")

        # ── FAISS index ──
        print(f"Loading FAISS index from {self.index_dir}...")
        self.faiss_index = faiss.read_index(f"{self.index_dir}/index.faiss")
        self.metadata = []
        with open(f"{self.index_dir}/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"✅ FAISS index ready ({len(self.metadata):,} vectors)")

        # ── Extractive QA model ──
        print("Loading extractive QA model...")
        qa_path = "/index/qa-model-finetuned" if os.path.exists("/index/qa-model-finetuned/model.safetensors") else "/models/qa-model"
        print(f"Loading QA model from {qa_path}...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device
        self._extract_answer("warmup question", "warmup context for the model")
        print("✅ Extractive QA model ready")

    def _extract_answer(self, question: str, context: str) -> dict:
        import torch
        inputs = self.qa_tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
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

        import torch.nn.functional as F
        start_prob = float(F.softmax(start_logits, dim=0)[start_idx])
        end_prob = float(F.softmax(end_logits, dim=0)[end_idx])
        score = round(start_prob * end_prob, 4)

        return {"answer": answer, "score": score, "start": start_idx, "end": end_idx}

    def _postprocess(self, answer: str, query: str, source_text: str) -> str:
        import re
        INDIC_DIGITS = {
            '०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9',
            '০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9',
            '૦':'0','૧':'1','૨':'2','૩':'3','૪':'4','૫':'5','૬':'6','૭':'7','૮':'8','૯':'9',
            '੦':'0','੧':'1','੨':'2','੩':'3','੪':'4','੫':'5','੬':'6','੭':'7','੮':'8','੯':'9',
            '୦':'0','୧':'1','୨':'2','୩':'3','୪':'4','୫':'5','୬':'6','৭':'7','৮':'8','٩':'9',
            '௦':'0','௧':'1','௨':'2','௩':'3','௪':'4','௫':'5','௬':'6','௭':'7','௮':'8','௯':'9',
            '౦':'0','౧':'1','౨':'2','౩':'3','౪':'4','౫':'5','౬':'6','౭':'7','౮':'8','౯':'9',
            '೦':'0','೧':'1','২':'2','೩':'3','೪':'4','೫':'5','೬':'6','೭':'7','೮':'8','೯':'9',
            '൦':'0','\u0d67':'1','\u0d68':'2','\u0d69':'3','\u0d6a':'4','\u0d6b':'5','\u0d6c':'6','\u0d6d':'7','\u0d6e':'8','\u0d6f':'9',
            '۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','६':'6','۷':'7','۸':'8','۹':'9',
        }
        for indic, arabic in INDIC_DIGITS.items():
            answer = answer.replace(indic, arabic)
            source_text = source_text.replace(indic, arabic)

        answer = re.sub(r'([$₹€£¥])\s+(\d)', r'\1\2', answer)
        answer = re.sub(r'\b(.{4,40})\s+\1\b', r'\1', answer)
        answer = answer.strip()
        answer = re.sub(r'^[,;:\-–—।\s]+', '', answer).strip()
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]

        words = answer.split()
        if len(words) < 4 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 40:
                    answer = sent.strip()
                    break
        return answer.strip()

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
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        if self.qa_device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        best = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": ""}
        
        for i in range(len(active_chunks)):
            s_logits = start_logits[i]
            e_logits = end_logits[i]
            
            start_idx = int(torch.argmax(s_logits))
            end_idx = int(torch.argmax(e_logits))
            
            if end_idx < start_idx:
                end_idx = start_idx
                
            tokens = inputs["input_ids"][i][start_idx:end_idx + 1]
            answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()
            
            start_prob = float(F.softmax(s_logits, dim=0)[start_idx])
            end_prob = float(F.softmax(e_logits, dim=0)[end_idx])
            score = round(start_prob * end_prob, 4)
            
            if score > best["score"] and answer:
                best = {
                    "answer": answer,
                    "score": score,
                    "chunk_idx": i,
                    "source_text": active_chunks[i]["text"],
                }
                
        if best["score"] >= 0 and best["answer"]:
            best["answer"] = self._postprocess(
                best["answer"], question, best["source_text"]
            )
        else:
            best = {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": ""}
            
        return best

    def _embed(self, text: str):
        import numpy as np
        query_text = f"query: {text}" if self.use_prefixes else text
        vec = self.embed_model.encode([query_text], normalize_embeddings=True)[0]
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _search(self, query_vec, k: int):
        import numpy as np
        qv = query_vec.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        scores, ids = self.faiss_index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            results.append((self.metadata[idx], float(score)))
        return results

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
        import time
        t0 = time.perf_counter()
        qvec = self._embed(query)
        t1 = time.perf_counter()
        candidates = self._search(qvec, self.RERANK_TOP_N)
        t2 = time.perf_counter()
        chunks = self._hybrid_rerank(query, candidates)[:self.TOP_K]
        t3 = time.perf_counter()
        return chunks, {
            "embed_ms":  round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": round((t3 - t2) * 1000, 2),
        }

    def _check_unsafe(self, query: str) -> bool:
        import re
        pattern = re.compile(
            r"\bhow to (make|build) (a )?(bomb|weapon|explosive)\b"
            r"|\bself[- ]?harm\b|\bhack (into|someone)\b", re.IGNORECASE
        )
        return bool(pattern.search(query))

    def _decline(self, query, reason, timings):
        msgs = {
            "unsafe_input":           "I can't help with that request.",
            "off_topic":              "That question is outside the knowledge base scope.",
            "low_retrieval_confidence": "I don't have enough grounded information.",
            "no_retrieval_results":   "Couldn't find anything relevant.",
            "low_qa_confidence":      "Couldn't extract a confident answer from the retrieved context.",
        }
        return {
            "query": query, "answer": msgs.get(reason, "Unable to answer."),
            "sources": [], "confidence": 0.0, "grounded": False,
            "guardrail_triggered": reason, "timings_ms": timings,
        }

    @modal.method()
    def rebuild_index_modal(self, use_prefixes: bool = True):
        import os
        import json
        import numpy as np
        import faiss
        
        sys.path.insert(0, "/root")
        from src.chunking.strategies import chunk_all_strategies
        
        print("Loading passages from volume...")
        passages = []
        with open("/index/passages.jsonl", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
        print(f"Loaded {len(passages)} passages.")
        
        def embed_fn(texts: list[str]) -> np.ndarray:
            processed_texts = [f"passage: {t}" if use_prefixes else t for t in texts]
            embs = self.embed_model.encode(processed_texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(embs, dtype=np.float32)
            
        print("Running chunking strategies...")
        all_chunks = []
        for row in passages:
            all_chunks.extend(chunk_all_strategies(row, embed_fn))
            
        print(f"Total chunks across all strategies: {len(all_chunks)}")
        
        texts = [c.text for c in all_chunks]
        print(f"Embedding {len(texts)} chunks on T4 GPU in batches...")
        batch_size = 256
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs.append(embed_fn(batch))
        vectors = np.vstack(vecs)
        print(f"Embeddings shape: {vectors.shape}")
        
        # Build FAISS HNSW
        dim = vectors.shape[1]
        print("Building HNSW flat index...")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(vectors.astype(np.float32))
        
        metadatas = [{
            "chunk_id": c.id,
            "text": c.text,
            "strategy": c.strategy,
            "lang": c.lang,
            "query_id": c.query_id,
            "source_passage_id": c.source_passage_id,
            "is_selected": c.is_selected,
            **c.extra,
        } for c in all_chunks]
        
        out_dir = "/index/temp_index"
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(index, f"{out_dir}/index.faiss")
        with open(f"{out_dir}/metadata.jsonl", "w", encoding="utf-8") as f:
            for m in metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"✅ Rebuild completed. New index saved in {out_dir}")

    @modal.method()
    def evaluate_queries_modal(self, query_data: list, index_dir: str = "/index", use_prefixes: bool = False):
        import faiss, json
        
        self.use_prefixes = use_prefixes
        print(f"Evaluating inside container: use_prefixes={self.use_prefixes}")
        
        if index_dir != self.index_dir:
            print(f"Loading FAISS index from override path: {index_dir}...")
            self.faiss_index = faiss.read_index(f"{index_dir}/index.faiss")
            self.metadata = []
            with open(f"{index_dir}/metadata.jsonl", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line))
            self.index_dir = index_dir
            print(f"✅ FAISS index loaded ({len(self.metadata):,} vectors)")
            
        results = []
        
        # Helper to calculate token F1 and EM
        def evaluate_answer(pred, gold_list):
            if not gold_list:
                return 0.0, 0.0
            
            def normalize(t):
                return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()
                
            pred_norm = normalize(pred)
            best_f1 = 0.0
            best_em = 0.0
            
            for gold in gold_list:
                gold_norm = normalize(gold)
                if pred_norm == gold_norm:
                    em = 1.0
                else:
                    em = 0.0
                best_em = max(best_em, em)
                
                # F1
                pred_toks = pred_norm.split()
                gold_toks = gold_norm.split()
                common = set(pred_toks) & set(gold_toks)
                if not pred_toks or not gold_toks:
                    f1 = 1.0 if pred_toks == gold_toks else 0.0
                else:
                    prec = len(common) / len(pred_toks)
                    rec = len(common) / len(gold_toks)
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                best_f1 = max(best_f1, f1)
                
            return best_em, best_f1
            
        for i, q in enumerate(query_data):
            query_text = q["query"]
            gold_answers = q["answers"]
            lang = q["lang"]
            
            chunks, ret_timings = self._retrieve(query_text)
            
            triggered = None
            top_score = chunks[0]["score"] if chunks else 0.0
            if top_score < self.OFF_TOPIC_THRESHOLD:
                triggered = "off_topic"
            elif not chunks or top_score < self.MIN_RETRIEVAL_SCORE:
                triggered = "low_retrieval_confidence"
                
            pred_answer = ""
            if not triggered:
                best = self._extract_best_answer(query_text, chunks)
                if best["score"] < self.MIN_QA_SCORE or not best["answer"]:
                    triggered = "low_qa_confidence"
                else:
                    pred_answer = best["answer"]
            else:
                pred_answer = self._decline(query_text, triggered, {})["answer"]
                
            gold_qid = q["query_id"]
            retrieved_qids = [c["query_id"] for c in chunks]
            
            recall_1 = 1.0 if retrieved_qids and retrieved_qids[0] == gold_qid else 0.0
            recall_5 = 1.0 if gold_qid in retrieved_qids[:5] else 0.0
            recall_10 = 1.0 if gold_qid in retrieved_qids[:10] else 0.0
            
            em, f1 = evaluate_answer(pred_answer, gold_answers)
            
            res = {
                "query_id": q["query_id"],
                "lang": lang,
                "em": em,
                "f1": f1,
                "recall_1": recall_1,
                "recall_5": recall_5,
                "recall_10": recall_10,
                "guardrail_triggered": triggered
            }
            results.append(res)
            
        return results

@app.local_entrypoint()
def main(action: str = "eval", index_dir: str = "/index", use_prefixes: str = "false"):
    rag = EvalVoiceRAG()
    
    if action == "rebuild":
        print("Starting remote index rebuild...")
        rag.rebuild_index_modal.remote(use_prefixes=(use_prefixes.lower() == "true"))
        print("Remote index rebuild completed.")
        return
        
    print("Loading local passages.jsonl...")
    rows = []
    with open("data/processed/passages.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            
    lang_passages = defaultdict(list)
    for r in rows:
        lang_passages[r["lang"]].append(r)
        
    import random
    random.seed(42)
    query_data = []
    for lang, passes in lang_passages.items():
        sampled = random.sample(passes, min(len(passes), 15))
        for s in sampled:
            query_data.append({
                "query_id": s["query_id"],
                "query": s["query"],
                "answers": s["answers"],
                "lang": s["lang"]
            })
            
    print(f"Sampled {len(query_data)} queries for evaluation.")
    print(f"Running evaluation with index_dir={index_dir} and use_prefixes={use_prefixes}...")
    
    results = rag.evaluate_queries_modal.remote(
        query_data,
        index_dir=index_dir,
        use_prefixes=(use_prefixes.lower() == "true")
    )
    
    print("\n=== Evaluation Results ===")
    total = len(results)
    overall_em = sum(r["em"] for r in results) / total
    overall_f1 = sum(r["f1"] for r in results) / total
    overall_rec1 = sum(r["recall_1"] for r in results) / total
    overall_rec5 = sum(r["recall_5"] for r in results) / total
    overall_rec10 = sum(r["recall_10"] for r in results) / total
    overall_declined = sum(1 for r in results if r["guardrail_triggered"]) / total
    
    print(f"Overall EM: {overall_em:.2%}")
    print(f"Overall F1: {overall_f1:.2%}")
    print(f"Overall Recall@1: {overall_rec1:.2%}")
    print(f"Overall Recall@5: {overall_rec5:.2%}")
    print(f"Overall Recall@10: {overall_rec10:.2%}")
    print(f"Overall Declined: {overall_declined:.2%}")
    
    lang_metrics = defaultdict(list)
    for r in results:
        lang_metrics[r["lang"]].append(r)
        
    for lang, res_list in sorted(lang_metrics.items()):
        em = sum(r["em"] for r in res_list) / len(res_list)
        f1 = sum(r["f1"] for r in res_list) / len(res_list)
        rec5 = sum(r["recall_5"] for r in res_list) / len(res_list)
        rec10 = sum(r["recall_10"] for r in res_list) / len(res_list)
        declined = sum(1 for r in res_list if r["guardrail_triggered"]) / len(res_list)
        print(f"  {lang}: EM={em:.2%} F1={f1:.2%} Recall@5={rec5:.2%} Recall@10={rec10:.2%} Declined={declined:.2%}")
