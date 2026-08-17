"""
Modal deployment for Voice RAG — Extractive QA (no generative LLM).
Version: 3.0.0 (Reindexed with query+answer chunks, 58k vectors)
"""
import modal

# ── Image ─────────────────────────────────────────────────────────────────────
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
        # xlm-roberta-base-squad2: multilingual, handles all 13 Indic langs, ~1.1GB
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


@app.cls(
    gpu="T4",
    scaledown_window=300,
    min_containers=1,
    volumes={"/index": volume},
)
class VoiceRAG:

    @modal.enter()
    def load(self):
        import os, torch
        from sentence_transformers import SentenceTransformer
        from transformers import (
            AutoTokenizer, AutoModelForQuestionAnswering, pipeline
        )
        import faiss, json

        device = 0 if torch.cuda.is_available() else -1
        print(f"Device: {'cuda' if device == 0 else 'cpu'}")

        # ── Embedding model ──
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)
        print("✅ Embedding model ready")

        # ── FAISS index ──
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index("/index/index.faiss")
        self.metadata = []
        with open("/index/metadata.jsonl", encoding="utf-8") as f:
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
        # Warmup pass to absorb JIT cost and cuda memory allocation
        self._extract_best_answer("warmup question", [{"text": "warmup context for the model"}])
        print("✅ Extractive QA model ready")

        # Config from Modal secrets
        self.OFF_TOPIC_THRESHOLD      = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.70"))
        self.MIN_RETRIEVAL_SCORE      = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.65"))
        self.MIN_QA_SCORE             = float(os.getenv("MIN_QA_SCORE", "0.10"))
        self.MIN_ANSWER_RELEVANCE     = float(os.getenv("MIN_ANSWER_RELEVANCE", "0.20"))
        self.TOP_K                    = int(os.getenv("TOP_K", "10"))
        self.RERANK_TOP_N             = int(os.getenv("RERANK_TOP_N", "50"))
        self.SARVAM_KEY               = os.getenv("SARVAM_API_KEY", "")
        self.SARVAM_URL               = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")

        # Indic + English stopwords for BM25 filtering
        self.STOPWORDS = set(
            # Hindi
            "का के की है में को और से हैं पर यह था थी थे "
            "इस कि एक भी ने जो वह हो तो कर इसके लिए अपने "
            "होता करने उनके साथ अगर अन्य कुछ तक जब "
            # English
            "the a an is are was were be been being have has had "
            "do does did will would shall should may might can could "
            "i me my we our you your he him his she her it its they them their "
            "what which who whom this that these those am "
            "in on at to for with from by of and or not no nor "
            "if but so than too very as how when where why all each every "
            # Common Urdu/Hindi
            "کا کی کے ہے میں "
            .split()
        )

    # ── Extractive QA ─────────────────────────────────────────────────────────

    def _extract_answer(self, question: str, context: str) -> dict:
        """
        Run XLM-RoBERTa QA over (question, context).
        Returns {"answer": str, "score": float, "start": int, "end": int}
        Score is the model's confidence in the extracted span.
        This is the core of why we can hit sub-200ms on ALL percentiles:
        one forward pass, no generation, no sampling.
        """
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

        # Clamp end to be >= start (edge case)
        if end_idx < start_idx:
            end_idx = start_idx

        tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()

        # Confidence: softmax over start * softmax over end (joint probability)
        import torch.nn.functional as F
        start_prob = float(F.softmax(start_logits, dim=0)[start_idx])
        end_prob = float(F.softmax(end_logits, dim=0)[end_idx])
        score = round(start_prob * end_prob, 4)

        return {"answer": answer, "score": score, "start": start_idx, "end": end_idx}

    def _postprocess(self, answer: str, query: str, source_text: str) -> str:
        """Pure string post-processing — zero latency cost."""
        import re, unicodedata

        # 1. Normalize ALL Indic script digits to Arabic numerals
        INDIC_DIGITS = {
            '०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9',
            '০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9',
            '૦':'0','૧':'1','૨':'2','૩':'3','૪':'4','૫':'5','૬':'6','૭':'7','૮':'8','૯':'9',
            '੦':'0','੧':'1','੨':'2','੩':'3','੪':'4','੫':'5','੬':'6','੭':'7','੮':'8','੯':'9',
            '୦':'0','୧':'1','୨':'2','୩':'3','୪':'4','୫':'5','୬':'6','୭':'7','୮':'8','୯':'9',
            '௦':'0','௧':'1','௨':'2','௩':'3','௪':'4','௫':'5','௬':'6','௭':'7','௮':'8','௯':'9',
            '౦':'0','౧':'1','౨':'2','౩':'3','౪':'4','౫':'5','౬':'6','౭':'7','౮':'8','౯':'9',
            '೦':'0','೧':'1','೨':'2','೩':'3','೪':'4','೫':'5','೬':'6','೭':'7','೮':'8','೯':'9',
            '൦':'0','൧':'1','൨':'2','൩':'3','൪':'4','൫':'5','൬':'6','൭':'7','൮':'8','൯':'9',
            '۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','۶':'6','۷':'7','۸':'8','۹':'9',
        }
        for indic, arabic in INDIC_DIGITS.items():
            answer = answer.replace(indic, arabic)
            source_text = source_text.replace(indic, arabic)

        # 2. Fix currency symbol spacing
        answer = re.sub(r'([$₹€£¥])\s+(\d)', r'\1\2', answer)

        # 3. Remove duplicate repeated phrases
        answer = re.sub(r'\b(.{4,40})\s+\1\b', r'\1', answer)

        # 4. Clean partial sentence start (strip leading punctuation, capitalize)
        answer = answer.strip()
        answer = re.sub(r'^[,;:\-–—।\s]+', '', answer).strip()
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]

        # 5. Expand very short answers (<4 words) using source sentence
        words = answer.split()
        if len(words) < 4 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 40:
                    answer = sent.strip()
                    break

        # 6. If English query but answer is entirely non-ASCII Indic script,
        #    keep it as-is (the passage language determines answer language —
        #    we can't transliterate without a model)
        return answer.strip()

    def _extract_best_answer(self, question: str, chunks: list) -> dict:
        """
        Run extractive QA over retrieved chunks in a parallel batch forward pass,
        pick the highest-scoring span across all chunks, then post-process.
        """
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

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _embed(self, text: str):
        vec = self.embed_model.encode([text], normalize_embeddings=True)[0]
        # Explicitly re-normalize to guard against any float precision drift
        import numpy as np
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _search(self, query_vec, k: int):
        import numpy as np
        qv = query_vec.astype("float32").reshape(1, -1)
        # Re-normalize query vector before inner product search
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
        """Remove stopwords from token list for cleaner BM25 matching."""
        return [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]

    def _hybrid_rerank(self, query: str, candidates: list):
        from rank_bm25 import BM25Okapi
        if not candidates: return []
        # Filter stopwords from both corpus and query for BM25
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

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, audio_bytes: bytes, lang=None):
        import time, httpx
        LANG_MAP = {
            "as":"as-IN","bn":"bn-IN","gu":"gu-IN","hi":"hi-IN",
            "kn":"kn-IN","ml":"ml-IN","mr":"mr-IN","ne":"ne-IN",
            "or":"od-IN","pa":"pa-IN","ta":"ta-IN","te":"te-IN","ur":"ur-IN",
        }
        bcp47 = LANG_MAP.get(lang) if lang else None
        headers = {"api-subscription-key": self.SARVAM_KEY}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "saaras:v3", "mode": "transcribe"}
        if bcp47: data["language_code"] = bcp47
        t0 = time.perf_counter()
        r = httpx.post(self.SARVAM_URL, headers=headers, files=files, data=data, timeout=10)
        r.raise_for_status()
        stt_ms = round((time.perf_counter() - t0) * 1000, 2)
        result = r.json()
        return {
            "transcript": result.get("transcript", ""),
            "language_detected": result.get("language_code"),
            "latency_ms": stt_ms,
        }

    # ── Guardrails ────────────────────────────────────────────────────────────

    def _check_unsafe(self, query: str) -> bool:
        import re
        pattern = re.compile(
            r"\bhow to (make|build) (a )?(bomb|weapon|explosive)\b"
            r"|\bself[- ]?harm\b|\bhack (into|someone)\b", re.IGNORECASE
        )
        return bool(pattern.search(query))

    def _token_overlap(self, a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta) if ta else 0.0

    # ── Decline helper ────────────────────────────────────────────────────────

    def _decline(self, query, reason, timings, transcript=None):
        msgs = {
            "unsafe_input":           "I can't help with that request.",
            "off_topic":              "That question is outside the knowledge base scope.",
            "low_retrieval_confidence": "I don't have enough grounded information.",
            "no_retrieval_results":   "Couldn't find anything relevant.",
            "low_qa_confidence":      "Couldn't extract a confident answer from the retrieved context.",
            "low_answer_relevance":   "The retrieved context doesn't appear relevant to your question.",
            "stt_failed":             "Couldn't transcribe audio -- please retry.",
        }
        return {
            "query": query, "transcript": transcript,
            "answer": msgs.get(reason, "Unable to answer."),
            "sources": [], "confidence": 0.0, "grounded": False,
            "guardrail_triggered": reason, "timings_ms": timings,
            "lang_detected": None,
        }

    # ── Main query pipeline ───────────────────────────────────────────────────

    def _run_query(self, query: str) -> dict:
        import time
        timings = {}
        t_start = time.perf_counter()

        # Guardrail 1 — unsafe input (regex, ~0ms)
        if self._check_unsafe(query):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "unsafe_input", timings)

        # Retrieval — embed + FAISS + hybrid rerank
        chunks, ret_timings = self._retrieve(query)
        timings.update(ret_timings)

        # Guardrail 2 — off-topic (top retrieval score too low)
        top_score = chunks[0]["score"] if chunks else 0.0
        if top_score < self.OFF_TOPIC_THRESHOLD:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "off_topic", timings)

        # Guardrail 3 — retrieval confidence
        if not chunks or top_score < self.MIN_RETRIEVAL_SCORE:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "low_retrieval_confidence", timings)

        # Extractive QA — single forward pass per chunk, pick best span
        t_qa0 = time.perf_counter()
        best = self._extract_best_answer(query, chunks)
        timings["qa_ms"] = round((time.perf_counter() - t_qa0) * 1000, 2)

        # Guardrail 4 — QA confidence (model wasn't sure about any span)
        if best["score"] < self.MIN_QA_SCORE or not best["answer"]:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "low_qa_confidence", timings)

        # Guardrail 5 — Query-answer semantic relevance
        # Catch cases where QA confidently extracts from an irrelevant passage
        import numpy as np
        answer_vec = self.embed_model.encode([best["answer"]], normalize_embeddings=True)[0]
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        relevance = float(np.dot(query_vec.astype(np.float32), answer_vec.astype(np.float32)))
        if relevance < self.MIN_ANSWER_RELEVANCE:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "low_answer_relevance", timings)

        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "query": query,
            "transcript": None,
            "answer": best["answer"],
            "sources": [
                {"text": c["text"], "score": c["score"],
                 "lang": c["lang"], "strategy": c["strategy"]}
                for c in chunks
            ],
            "confidence": round(best["score"], 4),
            "grounded": True,
            "guardrail_triggered": None,
            "timings_ms": timings,
            "lang_detected": None,
        }

    # ── FastAPI app ───────────────────────────────────────────────────────────

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.middleware.cors import CORSMiddleware
        import torch, time

        api = FastAPI(title="Voice RAG — Indic MSMARCO (Extractive QA)")
        api.add_middleware(
            CORSMiddleware, allow_origins=["*"],
            allow_methods=["*"], allow_headers=["*"]
        )

        @api.get("/debug-index")
        def debug_index():
            import numpy as np
            # Check metadata count vs index count
            n_meta = len(self.metadata)
            n_vecs = self.faiss_index.ntotal
            # Search with a zero vector to see raw scores
            q = np.zeros((1, 768), dtype=np.float32)
            q[0, 0] = 1.0
            scores, ids = self.faiss_index.search(q, 3)
            # Check first and last metadata entries
            first_meta = {k: v for k, v in list(self.metadata[0].items()) if k != "text"}
            last_meta = {k: v for k, v in list(self.metadata[-1].items()) if k != "text"}
            return {
                "n_metadata": n_meta,
                "n_vectors": n_vecs,
                "match": n_meta == n_vecs,
                "metric_type": self.faiss_index.metric_type,
                "test_scores": scores[0].tolist(),
                "test_ids": ids[0].tolist(),
                "first_metadata": first_meta,
                "last_metadata": last_meta,
            }

        @api.get("/debug-qa")
        def debug_qa(query: str = "हिरलूम टमाटर क्या है", context: str = "हिरलूम टमाटर एक पुरानी किस्म है जो खुले परागण से उगाई जाती है।"):
            result = self._extract_answer(query, context)
            chunks, _ = self._retrieve(query)
            chunk_results = []
            for c in chunks[:3]:
                r = self._extract_answer(query, c["text"])
                chunk_results.append({"chunk_lang": c["lang"], "chunk_text": c["text"][:100], "answer": r["answer"], "score": r["score"]})
            return {"direct_test": result, "top_chunks": chunk_results}

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "index_size": len(self.metadata),
                "gpu": torch.cuda.is_available(),
                "model": "xlm-roberta-base-squad2 (extractive QA)",
                "embed": "multilingual-e5-base",
            }

        @api.post("/query")
        async def text_query(query: str = Form(...)):
            return self._run_query(query)

        @api.post("/voice-query")
        async def voice_query(
            file: UploadFile = File(...),
            language_code: str = Form(None)
        ):
            t_start = time.perf_counter()
            try:
                audio_bytes = await file.read()
                stt = self._transcribe(audio_bytes, language_code)
            except Exception as e:
                print(f"STT error: {e}")
                return self._decline(
                    "<audio>", "stt_failed",
                    {"total_ms": round((time.perf_counter() - t_start) * 1000, 2)}
                )
            resp = self._run_query(stt["transcript"])
            resp["transcript"] = stt["transcript"]
            resp["lang_detected"] = stt.get("language_detected")
            resp["timings_ms"]["stt_ms"] = stt["latency_ms"]
            return resp

        return api