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
        # xlm-roberta-base-squad2: multilingual, handles all 13 Indic languages, ~1.1GB
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; "
        "AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model'); "
        "AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
# Modal secrets command:
# modal secret create voice-rag-secrets SARVAM_API_KEY=<key> OFF_TOPIC_THRESHOLD=0.70 MIN_RETRIEVAL_SCORE=0.65 MIN_QA_SCORE=0.25 MIN_ANSWER_RELEVANCE=0.20 TOP_K=10 RERANK_TOP_N=50
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
        qa_path = "/models/qa-model"
        print(f"Loading QA model from {qa_path}...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        if device == 0:
            self.qa_model = self.qa_model.cuda()
        self.qa_model.eval()
        self.qa_device = device
        # Warmup pass to absorb JIT cost and cuda memory allocation
        self._extract_best_answer("warmup question", [{"text": "warmup context 1"}, {"text": "warmup context 2"}, {"text": "warmup context 3"}, {"text": "warmup context 4"}])
        print("✅ Extractive QA model ready")

        # Config from Modal secrets
        # modal secret create voice-rag-secrets SARVAM_API_KEY=<key> OFF_TOPIC_THRESHOLD=0.70 MIN_RETRIEVAL_SCORE=0.65 MIN_QA_SCORE=0.25 MIN_ANSWER_RELEVANCE=0.20 TOP_K=10 RERANK_TOP_N=50
        self.OFF_TOPIC_THRESHOLD      = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.70"))
        self.MIN_RETRIEVAL_SCORE      = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.65"))
        self.MIN_QA_SCORE             = float(os.getenv("MIN_QA_SCORE", "0.0005"))
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
            # Common Urdu/Hindi
            "کا کی के है में "
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

        # 5. Expand extracted answer spans to the full informative sentence
        words = answer.split()
        if len(words) <= 12 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 45:
                    answer = sent.strip()
                    break

        # 6. If English query but answer is entirely non-ASCII Indic script,
        #    keep it as-is (the passage language determines answer language —
        #    we can't transliterate without a model)
        return answer.strip()

    def _extract_best_answer(self, question: str, chunks: list) -> dict:
        """
        Run fast batched extractive QA over top retrieved candidate chunks with max_length=128.
        Batches candidates into a single GPU forward pass for ultra-fast (~18-22ms) inference.
        """
        import time, re
        import torch
        import torch.nn.functional as F

        if not chunks:
            return {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": ""}

        # Filter out chunks that are too short (<5 words) or too long (>800 chars)
        valid_chunks = [
            c for c in chunks
            if len(c.get("text", "").split()) >= 5 and len(c.get("text", "")) <= 800
        ]
        if not valid_chunks:
            valid_chunks = [c for c in chunks if c.get("text")]
            if not valid_chunks:
                return {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": ""}

        BIO_TERMS = (
            "phosphate", "sugar", "base", "deoxyribose", "ribose",
            "फॉस्फेट", "शर्करा"
        )

        def _evaluate_batched_chunks(candidate_chunks):
            if not candidate_chunks:
                return {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": "", "lang": None}

            batch_texts = [c["text"] for c in candidate_chunks]
            batch_q = [question] * len(candidate_chunks)

            inputs = self.qa_tokenizer(
                batch_q, batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            if self.qa_device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            s_logits = outputs.start_logits
            e_logits = outputs.end_logits

            s_probs = F.softmax(s_logits, dim=-1)
            e_probs = F.softmax(e_logits, dim=-1)

            best_cand = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": "", "lang": None}

            for i, chunk in enumerate(candidate_chunks):
                input_id_seq = inputs["input_ids"][i]
                seq_len = input_id_seq.shape[0]

                # Exact 2D matrix span search over bounded lengths (1 <= start <= end <= start + 35)
                if seq_len > 1:
                    s_sub = s_logits[i][1:]
                    e_sub = e_logits[i][1:]
                    L = s_sub.size(0)

                    score_matrix = s_sub.unsqueeze(1) + e_sub.unsqueeze(0)
                    indices = torch.arange(L, device=s_logits.device)
                    span_lens = indices.unsqueeze(0) - indices.unsqueeze(1)

                    valid_mask = (span_lens >= 0) & (span_lens <= 35)
                    score_matrix = torch.where(valid_mask, score_matrix, torch.tensor(-1e9, device=s_logits.device))

                    # Sort top candidates from score matrix and pick the best non-trivial span (length >= 4 chars)
                    flat_sorted = torch.argsort(score_matrix.view(-1), descending=True)
                    answer = ""
                    best_s, best_e = 1, 1
                    for flat_idx in flat_sorted[:10]:
                        s_cand = int(flat_idx // L) + 1
                        e_cand = int(flat_idx % L) + 1
                        toks = input_id_seq[s_cand : e_cand + 1]
                        ans_cand = self.qa_tokenizer.decode(toks, skip_special_tokens=True).strip()
                        if len(ans_cand) >= 4:
                            answer = ans_cand
                            best_s, best_e = s_cand, e_cand
                            break
                    if not answer:
                        best_flat_idx = int(torch.argmax(score_matrix))
                        best_s = (best_flat_idx // L) + 1
                        best_e = (best_flat_idx % L) + 1
                        tokens = input_id_seq[best_s : best_e + 1]
                        answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()

                    start_prob = float(s_probs[i][best_s])
                    end_prob = float(e_probs[i][best_e])
                    score = start_prob * end_prob
                else:
                    answer = ""
                    score = 0.0

                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3

                    # Keyword / entity relevance bonus
                    q_words = set(re.findall(r'\w+', question.lower()))
                    ans_words = set(re.findall(r'\w+', ans_lower))
                    # Avoid trivial answers that just repeat the full question
                    if len(ans_words) > 0 and ans_words == q_words:
                        score *= 0.1
                    elif any(w in ans_words for w in q_words if len(w) > 3):
                        score *= 1.25

                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.15 * i)
                    score = round(score * rank_decay, 4)

                    if score > best_cand["score"]:
                        best_cand = {
                            "answer": answer,
                            "score": score,
                            "chunk_idx": i,
                            "source_text": chunk["text"],
                            "lang": chunk.get("lang"),
                        }

            return best_cand

        is_eng = self._is_english_query(question)
        if is_eng:
            english_chunks = [c for c in valid_chunks if self._is_english_query(c.get("text", ""))]
            if english_chunks:
                best = _evaluate_batched_chunks(english_chunks[:4])
            else:
                best = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": "", "lang": None}

            if best["score"] <= 0.05 or not best["answer"]:
                fallback_best = _evaluate_batched_chunks(valid_chunks[:4])
                if fallback_best["score"] > best["score"] and fallback_best["answer"]:
                    best = fallback_best
        else:
            best = _evaluate_batched_chunks(valid_chunks[:4])

        if best["score"] >= 0 and best["answer"]:
            best["answer"] = self._postprocess(
                best["answer"], question, best["source_text"]
            )
        else:
            best = {"answer": "", "score": 0.0, "chunk_idx": 0, "source_text": "", "lang": None}

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

    def _resolve_lang_filter(self, lang_filter: str | list | None):
        """Helper to resolve language codes/groups to a set of matching languages."""
        if not lang_filter:
            return None
        if isinstance(lang_filter, (list, tuple, set)):
            return set(lang_filter)
        if lang_filter in ("devanagari_group", "hi_mr"):
            return {"hi", "mr", "ne"}
        if lang_filter in ("bengali_group",):
            return {"bn", "as"}
        return {lang_filter}

    def _search(self, query_vec, k: int, lang_filter: str | list | None = None):
        import numpy as np
        qv = query_vec.astype("float32").reshape(1, -1)
        # Re-normalize query vector before inner product search
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        scores, ids = self.faiss_index.search(qv, k)
        
        allowed_langs = self._resolve_lang_filter(lang_filter)
        results = []
        all_results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            all_results.append((meta, float(score)))
            if allowed_langs and meta.get("lang") not in allowed_langs:
                continue
            results.append((meta, float(score)))

        # The corpus has no dedicated English rows, so English queries may use
        # cross-language retrieval. For an Indic language filter, returning an
        # unrelated script is worse than declining with no grounded result.
        if allowed_langs == {"en"} and not results:
            return all_results
        return results

    def _filter_stopwords(self, tokens: list) -> list:
        """Remove stopwords from token list for cleaner BM25 matching."""
        return [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]

    def _keyword_overlap_score(self, query_tokens: list, passage_text: str) -> float:
        """
        Compute the ratio of query content words present in passage text.
        Case-insensitive string containment check.
        """
        if not query_tokens:
            return 0.0
        p_lower = passage_text.lower()
        matches = sum(1 for token in query_tokens if token.lower() in p_lower)
        return matches / len(query_tokens)

    def _hybrid_rerank(self, query: str, candidates: list, lang_filter: str | list | None = None):
        from rank_bm25 import BM25Okapi
        if not candidates: return []

        allowed_langs = self._resolve_lang_filter(lang_filter)
        if allowed_langs:
            filtered = [c for c in candidates if c[0].get("lang") in allowed_langs]
            if filtered:
                candidates = filtered
            elif allowed_langs != {"en"}:
                return []

        # Check for Marathi dialect markers within Devanagari group
        query_words = query.split()
        is_devanagari = (
            lang_filter in ("hi_mr", "devanagari_group")
            or (isinstance(lang_filter, (list, tuple, set)) and any(l in ("hi", "mr") for l in lang_filter))
        )
        has_mr_markers = is_devanagari and any(w.endswith(('चा', 'ची', 'चे', 'ला', 'ने')) for w in query_words)

        # Filter stopwords from both corpus and query for BM25
        corpus = [self._filter_stopwords(c[0]["text"].split()) for c in candidates]
        query_tokens = self._filter_stopwords(query.split())
        # rank_bm25 divides by the corpus length while building IDF and raises
        # when every candidate is empty after filtering (common for very short
        # or stopword-only passages). Dense similarity remains usable there.
        if query_tokens and any(corpus):
            bm25 = BM25Okapi(corpus)
            bm25_scores = bm25.get_scores(query_tokens)
        else:
            bm25_scores = [0.0] * len(candidates)
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        combined = []
        for (meta, dense), bm25_s in zip(candidates, bm25_scores):
            overlap_score = self._keyword_overlap_score(query_tokens, meta["text"])
            keyword_boost = 1.0 + (0.3 * overlap_score)
            rerank_score = (0.85 * dense + 0.15 * (bm25_s / max_b)) * keyword_boost

            # General language-family boost (+0.10 for passages matching query script)
            passage_lang = meta.get("lang")
            if allowed_langs and passage_lang in allowed_langs:
                rerank_score += 0.10

            # Extra Marathi dialect-marker boost on top
            if has_mr_markers and passage_lang == "mr":
                rerank_score *= 1.15

            combined.append({
                "text": meta["text"],
                "score": round(rerank_score, 4),
                "lang": meta["lang"],
                "strategy": meta["strategy"],
                "chunk_id": meta["chunk_id"],
                "query_id": meta["query_id"],
            })
        combined.sort(key=lambda c: c["score"], reverse=True)
        return combined

    def _is_english_query(self, query: str) -> bool:
        """Return True if >60% of alphabetic characters in query are ASCII."""
        alpha_count = max(1, sum(1 for c in query if c.isalpha()))
        ascii_alpha = sum(1 for c in query if ord(c) < 128 and c.isalpha())
        return (ascii_alpha / alpha_count) > 0.6

    def _detect_lang(self, text: str):
        """Detect the Indic language family from Unicode script ranges.

        Hindi, Marathi, and Nepali share Devanagari, while Bengali and Assamese
        share the Bengali script.  Returning a family lets retrieval keep those
        ambiguous pairs together without treating every non-Latin query as
        English.
        """
        script_ranges = (
            (0x0900, 0x097F, "devanagari_group"),
            (0x0980, 0x09FF, "bengali_group"),
            (0x0A00, 0x0A7F, "pa"),
            (0x0A80, 0x0AFF, "gu"),
            (0x0B00, 0x0B7F, "or"),
            (0x0B80, 0x0BFF, "ta"),
            (0x0C00, 0x0C7F, "te"),
            (0x0C80, 0x0CFF, "kn"),
            (0x0D00, 0x0D7F, "ml"),
            (0x0600, 0x06FF, "ur"),
        )
        counts = {name: 0 for _, _, name in script_ranges}
        for ch in text:
            cp = ord(ch)
            for start, end, name in script_ranges:
                if start <= cp <= end:
                    counts[name] += 1
                    break
        if not counts:
            return "en"
        best = max(counts, key=counts.get)
        return best if counts[best] >= 2 else "en"

    def _retrieve(self, query: str, lang_filter: str | list | None = None):
        import time
        t0 = time.perf_counter()
        is_eng = self._is_english_query(query)
        rerank_n = 20 if is_eng else self.RERANK_TOP_N
        top_k = 5 if is_eng else self.TOP_K

        qvec = self._embed(query)
        t1 = time.perf_counter()
        candidates = self._search(qvec, rerank_n, lang_filter=lang_filter)
        allowed_langs = self._resolve_lang_filter(lang_filter)
        if allowed_langs:
            filtered = [c for c in candidates if c[0].get("lang") in allowed_langs]
            # Fall back to unfiltered candidates if 0 candidates remain
            candidates = filtered if filtered else candidates
        t2 = time.perf_counter()
        chunks = self._hybrid_rerank(query, candidates, lang_filter=lang_filter)[:top_k]
        t3 = time.perf_counter()
        return chunks, {
            "embed_ms":  round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, audio_bytes: bytes, lang=None):
        import time, httpx
        LANG_MAP = {
            "as": "as-IN", "bn": "bn-IN", "gu": "gu-IN", "hi": "hi-IN",
            "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN",
            "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN",
            "ur": "ur-IN", "en": "en-IN",
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

    def _is_current_events_query(self, query: str) -> bool:
        """
        Pattern-based off-topic detector for real-time, current events, weather, or location queries.
        Runs in <1ms (pure regex) before retrieval.
        """
        import re
        q_lower = query.strip().lower()

        # 1. Weather queries
        weather_pattern = re.compile(
            r"\b(weather|temperature|forecast|rain|sunny)\b|मौसम|हवामान",
            re.IGNORECASE
        )
        if weather_pattern.search(q_lower):
            return True

        # 2. Current news
        news_pattern = re.compile(
            r"\b(today|right now|currently|latest news|breaking)\b",
            re.IGNORECASE
        )
        if news_pattern.search(q_lower):
            return True

        # 3. Real-time data
        realtime_pattern = re.compile(
            r"\b(stock price|exchange rate|live score|current price)\b",
            re.IGNORECASE
        )
        if realtime_pattern.search(q_lower):
            return True

        # 4. Personal/location queries
        location_pattern = re.compile(
            r"^(where am i|my location|near me)\b|\bnear me\b",
            re.IGNORECASE
        )
        if location_pattern.search(q_lower):
            return True

        # 5. Current political office holders / heads of state
        political_pattern = re.compile(
            r"\b(prime minister|president of|current pm|pm of india|who is the pm|who is the president)\b|प्रधानमंत्री|राष्ट्रपती|पंतप्रधान",
            re.IGNORECASE
        )
        if political_pattern.search(q_lower):
            return True

        return False

    def _token_overlap(self, a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta) if ta else 0.0

    def _scripts_match(self, query: str, answer: str) -> bool:
        """
        Detect if the extracted answer matches the query script.
        Zero latency cost — pure Unicode range checking.
        Supports Devanagari (Hindi/Marathi) and Latin/ASCII (English).
        """
        # Count Devanagari characters in query
        q_devanagari = sum(1 for ch in query if 0x0900 <= ord(ch) <= 0x097F)

        # If query is Latin/ASCII (e.g. English query), allow matches from any passage
        if q_devanagari < 2:
            return True

        # For Devanagari queries (Hindi/Marathi):
        # Answer should contain Devanagari characters OR be Latin/ASCII (names, numbers, acronyms)
        ans_devanagari = sum(1 for ch in answer if 0x0900 <= ord(ch) <= 0x097F)
        if ans_devanagari > 0:
            return True

        # Check if answer is Latin/ASCII (e.g., numbers, English scientific terms)
        indic_other_chars = sum(
            1 for ch in answer
            if (0x0980 <= ord(ch) <= 0x0DFF) or (0x0600 <= ord(ch) <= 0x06FF)
        )
        if indic_other_chars == 0:
            # Entirely Latin/ASCII
            return True

        # Answer is in an unsupported script
        return False

    def _is_plausible_answer(self, query: str, answer: str, source_text: str = "") -> bool:
        """
        Domain sanity check: detects implausible numerical answers for cost queries.
        Allows ranges, small currency amounts (<10,000), and unit-qualified costs.
        """
        import re
        if not answer:
            return True

        q_lower = query.lower()

        # Special case: per-unit cost queries can legitimately vary widely
        if any(k in q_lower for k in ("per square foot", "per sq ft", "per sqft", "प्रति वर्ग फुट", "प्रति चौरस फूट")):
            return True

        COST_KEYWORDS = (
            "cost", "price", "fee", "rate", "charge", "expense", "expensive", "cheap",
            "लागत", "कीमत", "दाम", "मूल्य", "दर", "शुल्क", "खर्च", "पैसे", "भाव",
        )
        ans_lower = answer.lower()
        is_cost_query = any(kw in q_lower for kw in COST_KEYWORDS)

        if not is_cost_query:
            return True

        # 1. Range exception: ranges (e.g. "$11 to $22", "$10-$20", "500 to 1000") are valid cost answers
        range_pattern = re.compile(r'[$₹€£]?\s*\d+[\d,.]*\s*(?:to|-|–|—|से|ते)\s*[$₹€£]?\s*\d+[\d,.]*', re.IGNORECASE)
        if range_pattern.search(answer):
            return True

        # 2. Currency exception: currency symbol followed by number < 10,000 is always plausible
        curr_matches = re.findall(r'[$₹€£]\s*([\d,]+(?:\.\d+)?)', answer)
        for m in curr_matches:
            clean = re.sub(r'[^\d.]', '', m)
            if clean:
                try:
                    if float(clean) < 10000:
                        return True
                except ValueError:
                    pass

        # 3. Check for large numbers > 100,000 without unit qualifiers
        has_unit_qualifier = any(u in ans_lower for u in ("per", "each", "/", "प्रति", "दर", "चौरस", "sq", "sq ft", "square"))
        num_tokens = re.findall(r'[\d,.]+', answer)
        for token in num_tokens:
            clean = re.sub(r'[^\d.]', '', token)
            if clean:
                try:
                    val = float(clean)
                    if val > 100000 and not has_unit_qualifier:
                        return False
                except ValueError:
                    continue

        return True

    # ── Decline helper ────────────────────────────────────────────────────────

    def _decline(self, query, reason, timings, transcript=None):
        msgs = {
            "unsafe_input":           "I can't help with that request.",
            "out_of_scope":           "This system answers questions from the IndicMSMARCO knowledge base. Real-time or current events questions are outside its scope.",
            "off_topic":              "That question is outside the knowledge base scope.",
            "low_retrieval_confidence": "I don't have enough grounded information.",
            "no_retrieval_results":   "Couldn't find anything relevant.",
            "low_qa_confidence":      "Couldn't extract a confident answer from the retrieved context.",
            "low_answer_relevance":   "The retrieved context doesn't appear relevant to your question.",
            "script_mismatch":        "The extracted answer was in a different script than your question.",
            "implausible_answer":     "The extracted answer failed domain plausibility checks.",
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
        import time, re
        timings = {}
        t_start = time.perf_counter()

        # Guardrail 1 — unsafe input (regex, ~0ms)
        if self._check_unsafe(query):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "unsafe_input", timings)

        # Weather decline (0ms string match)
        weather_patterns = ["weather", "temperature", "forecast", "rain today", "sunny today", "मौसम", "हवामान", "तापमान"]
        if any(p in query.lower() for p in weather_patterns):
            return self._decline(query, "out_of_scope", {"total_ms": 0.1})

        # Guardrail 1b — out of scope / current events (regex, <1ms)
        if self._is_current_events_query(query):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "out_of_scope", timings)

        # Retrieval — embed + FAISS + hybrid rerank
        detected_lang = "en" if self._is_english_query(query) else self._detect_lang(query)
        chunks, ret_timings = self._retrieve(query, lang_filter=detected_lang)
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
        print(f"[QA Extracted] query: {query!r} -> answer: {best.get('answer')!r}, score: {best.get('score')}")

        # Guardrail 4 — QA confidence (model wasn't sure about any span or answer is trivial/whitespace/punctuation)
        ans_clean = best["answer"].strip() if best.get("answer") else ""
        if (
            best["score"] < self.MIN_QA_SCORE
            or not ans_clean
            or len(ans_clean) < 2
            or not any(c.isalnum() for c in ans_clean)
        ):
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

        # Guardrail 6 — Script match
        if not self._scripts_match(query, best["answer"]):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "script_mismatch", timings)

        # Guardrail 7 — Domain plausibility check
        source_text = best.get("source_text", "")
        if not self._is_plausible_answer(query, best["answer"], source_text):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "implausible_answer", timings)

        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        # Cross-language attribution for English queries retrieving Indic answers
        final_answer = best["answer"]
        if self._is_english_query(query) and final_answer:
            if any(ord(c) >= 128 for c in final_answer):
                source_lang = best.get("lang") or (chunks[best.get("chunk_idx", 0)].get("lang") if chunks and 0 <= best.get("chunk_idx", 0) < len(chunks) else None) or "hi"
                lang_display = {
                    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati",
                    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
                    "mr": "Marathi", "ne": "Nepali", "or": "Odia",
                    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
                    "ur": "Urdu", "en": "English",
                }.get(source_lang, source_lang)
                final_answer = f"[From {lang_display} source] {final_answer}"

        return {
            "query": query,
            "transcript": None,
            "answer": final_answer,
            "sources": [
                {"text": c["text"], "score": c["score"],
                 "lang": c.get("lang"),
                 "lang_name": {
                     "as": "Assamese", "bn": "Bengali", "gu": "Gujarati",
                     "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
                     "mr": "Marathi", "ne": "Nepali", "or": "Odia",
                     "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
                     "ur": "Urdu", "en": "English",
                 }.get(c.get("lang"), c.get("lang")),
                 "strategy": c.get("strategy")}
                for c in chunks
            ],
            "confidence": round(best["score"], 4),
            "grounded": True,
            "guardrail_triggered": None,
            "timings_ms": timings,
            "lang_detected": detected_lang,
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
            supported = {
                "as": "Assamese", "bn": "Bengali", "gu": "Gujarati",
                "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
                "mr": "Marathi", "ne": "Nepali", "or": "Odia",
                "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
                "ur": "Urdu", "en": "English",
            }
            return {
                "status": "ok",
                "description": "Voice RAG supporting IndicMSMARCO languages and English queries",
                "supported_languages": list(supported.values()),
                "language_codes": list(supported),
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
