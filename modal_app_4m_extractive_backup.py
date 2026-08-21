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
        "orjson>=3.10.0",
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
        "\"",
        # Bake NLI entailment model into image at build time
        # MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification; "
        "AutoTokenizer.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7').save_pretrained('/models/nli-model'); "
        "AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7').save_pretrained('/models/nli-model')"
        "\""
    )
    .add_local_dir("frontend", remote_path="/root/frontend")
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
# Modal secrets command:
# modal secret create voice-rag-secrets SARVAM_API_KEY=<key> OFF_TOPIC_THRESHOLD=0.70 MIN_RETRIEVAL_SCORE=0.65 MIN_QA_SCORE=0.25 MIN_ANSWER_RELEVANCE=0.20 TOP_K=10 RERANK_TOP_N=50
secrets = [modal.Secret.from_name("voice-rag-secrets")]

app = modal.App("voice-rag", image=image, secrets=secrets)


class MetadataStore:
    def __init__(self, meta_path: str):
        import time
        t0 = time.perf_counter()
        self.data = []
        try:
            import orjson
            with open(meta_path, "rb") as f:
                for line in f:
                    if line.strip():
                        self.data.append(orjson.loads(line))
        except Exception:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        self.size = len(self.data)
        print(f"Loaded {self.size:,} metadata entries in {time.perf_counter()-t0:.1f}s")

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = self.size + idx
        if idx < 0 or idx >= self.size:
            return {}
        return self.data[idx]


@app.cls(
    gpu="A10G",
    cpu=8.0,
    memory=16384,
    scaledown_window=420,
    min_containers=0,
    timeout=300,
    volumes={"/index": volume},
)
class VoiceRAG:

    @modal.enter()
    def load(self):
        import os, torch, shutil, time
        from sentence_transformers import SentenceTransformer
        from transformers import (
            AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, pipeline
        )
        import faiss, json

        device = 0 if torch.cuda.is_available() else -1
        print(f"Device: {'cuda' if device == 0 else 'cpu'}")

        # ── Copy from Volume to local disk for fast lookup ──
        volume.reload()
        local_index_path = "/tmp/index.faiss"
        local_meta_path = "/tmp/metadata.jsonl"
        vol_index_size = os.path.getsize("/index/index.faiss") if os.path.exists("/index/index.faiss") else 0
        local_index_size = os.path.getsize(local_index_path) if os.path.exists(local_index_path) else 0

        if not os.path.exists(local_index_path) or vol_index_size != local_index_size:
            print("Copying latest index from Volume to local disk...")
            t0 = time.perf_counter()
            shutil.copyfile("/index/index.faiss", local_index_path)
            shutil.copyfile("/index/metadata.jsonl", local_meta_path)
            print(f"Index and metadata copied in {time.perf_counter()-t0:.1f}s")

        # Set FAISS threads explicitly
        n_threads = os.cpu_count() or 8
        faiss.omp_set_num_threads(n_threads)
        print(f"FAISS using {faiss.omp_get_max_threads()} threads (os.cpu_count()={os.cpu_count()})")

        # ── Embedding model ──
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda" if device == 0 else "cpu")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)
        print("✅ Embedding model ready")

        # ── FAISS index (Loaded & Transferred to A10G GPU Tensor Memory) ──
        print("Loading FAISS index into RAM and GPU Tensor memory...")
        self.faiss_index = faiss.read_index(local_index_path)
        print(f"FAISS index loaded: {self.faiss_index.ntotal:,} vectors")
        self.gpu_vectors = None
        self.gpu_vector_err = None
        if device == 0:
            try:
                import numpy as np
                raw_bytes = faiss.vector_to_array(self.faiss_index.codes)
                raw_vecs = raw_bytes.view(np.float32).reshape(self.faiss_index.ntotal, self.faiss_index.d)
                self.gpu_vectors = torch.from_numpy(raw_vecs).cuda().half()
                print(f"✅ Transferred {self.faiss_index.ntotal:,} vectors to A10G Tensor Core memory (FP16: {self.gpu_vectors.element_size() * self.gpu_vectors.nelement() / (1024**3):.2f} GB)")
            except Exception as e:
                import traceback
                self.gpu_vector_err = f"{e}\n{traceback.format_exc()}"
                print(f"Fallback to CPU FAISS search: {self.gpu_vector_err}")

        # ── Fast In-Memory Metadata Store ──
        print("Loading metadata store into RAM...")
        self.metadata = MetadataStore(local_meta_path)

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

        # ── NLI Entailment model (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) ──
        print("Loading NLI entailment model...")
        nli_path = "/models/nli-model"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_path)
        if device == 0:
            self.nli_model = self.nli_model.cuda()
        self.nli_model.eval()
        self.nli_device = device
        # Warmup pass
        self._check_entailment("Premise text for warmup.", "Warmup question?", "Warmup answer.")
        print("✅ NLI entailment model ready")

        # Config from Modal secrets
        # modal secret create voice-rag-secrets SARVAM_API_KEY=<key> OFF_TOPIC_THRESHOLD=0.70 MIN_RETRIEVAL_SCORE=0.65 MIN_QA_SCORE=0.25 MIN_ANSWER_RELEVANCE=0.20 TOP_K=10 RERANK_TOP_N=50
        self.OFF_TOPIC_THRESHOLD      = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.70"))
        self.MIN_RETRIEVAL_SCORE      = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.65"))
        self.MIN_QA_SCORE             = float(os.getenv("MIN_QA_SCORE", "0.0005"))
        self.MIN_ANSWER_RELEVANCE     = float(os.getenv("MIN_ANSWER_RELEVANCE", "0.20"))
        self.TOP_K                    = int(os.getenv("TOP_K", "8"))
        self.RERANK_TOP_N             = int(os.getenv("RERANK_TOP_N", "35"))
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
                max_length=256,
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

                    # Intent-specific entity bonus
                    # 1. Location intent (where / कहाँ / कुठे / কোথায় / ఎక్కడ / எங்கு)
                    if any(w in question.lower() for w in ("where", "कहाँ", "कुठे", "কোথায়", "ఎక్కడ", "எங்கு", "ਕਿੱਥੇ")):
                        if any(term in ans_lower for term in ("जंगल", "देश", "प्रदेश", "forest", "mountain", "country", "city", "क्षेत्र", "प्रदेशात", "मध्ये", "இல்")):
                            score *= 1.35

                    # 2. Cost / numerical intent (cost / price / how much / कितना / खर्च / কত / ధర / விலை)
                    if any(w in question.lower() for w in ("cost", "price", "how much", "कितना", "खर्च", "दर", "दाम", "কত", "ధర", "விலை")):
                        if any(c.isdigit() for c in answer) or any(s in answer for s in ("$", "₹", "€", "£")):
                            score *= 1.40

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

        # Strict language-matching: feed only same-language chunks to QA
        is_eng = self._is_english_query(question)
        if is_eng:
            # For English queries, prefer English-text chunks
            english_chunks = [c for c in valid_chunks if self._is_english_query(c.get("text", ""))]
            if english_chunks:
                best = _evaluate_batched_chunks(english_chunks[:6])
            else:
                # No English chunks found — try all chunks but don't cross-lang fallback blindly
                best = _evaluate_batched_chunks(valid_chunks[:6])
        else:
            # For Hindi/Marathi: detect query lang and strictly filter
            q_lang = self._detect_lang(question)
            same_lang_chunks = [c for c in valid_chunks if c.get("lang") == q_lang]
            if same_lang_chunks:
                best = _evaluate_batched_chunks(same_lang_chunks[:6])
            else:
                best = _evaluate_batched_chunks(valid_chunks[:6])

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
            return {"hi", "mr"}
        if lang_filter in ("bengali_group",):
            return {"bn", "as"}
        return {lang_filter}

    def _search(self, query_vec, k: int, lang_filter: str | list | None = None):
        import torch, numpy as np
        qv = query_vec.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm

        if getattr(self, "gpu_vectors", None) is not None:
            with torch.no_grad():
                qv_t = torch.from_numpy(qv).cuda().half()
                sims = torch.matmul(self.gpu_vectors, qv_t.T).squeeze(1)
                top_scores, top_ids = torch.topk(sims, k=min(k, len(self.metadata)))
                scores_list = top_scores.float().cpu().numpy().tolist()
                ids_list = top_ids.cpu().numpy().tolist()
        else:
            scores, ids = self.faiss_index.search(qv, k)
            scores_list = scores[0].tolist()
            ids_list = ids[0].tolist()

        allowed_langs = self._resolve_lang_filter(lang_filter)
        results = []
        for score, idx in zip(scores_list, ids_list):
            idx = int(idx)
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if allowed_langs and meta.get("lang") not in allowed_langs:
                continue
            results.append((meta, float(score)))

        return results

    def _filter_stopwords(self, tokens: list) -> list:
        """Remove stopwords from token list for cleaner BM25 matching."""
        return [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 1]

    def _keyword_overlap_score(self, query_tokens: list, passage_text: str) -> float:
        """
        Compute morphological & lexical overlap using word containment and character trigrams.
        Handles inflected Indic root words accurately.
        """
        if not query_tokens:
            return 0.0
        p_lower = passage_text.lower()
        word_matches = sum(1 for token in query_tokens if token.lower() in p_lower)
        word_ratio = word_matches / len(query_tokens)

        # Character trigrams for root-word matching
        q_str = " ".join(query_tokens).lower()
        if len(q_str) >= 3 and len(p_lower) >= 3:
            q_tri = set(q_str[i:i+3] for i in range(len(q_str)-2))
            p_tri = set(p_lower[i:i+3] for i in range(len(p_lower)-2))
            tri_ratio = len(q_tri & p_tri) / max(1, len(q_tri))
        else:
            tri_ratio = 0.0

        return max(word_ratio, tri_ratio)

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
        corpus = [self._filter_stopwords(c[0].get("text", "").split()) for c in candidates]
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
            overlap_score = self._keyword_overlap_score(query_tokens, meta.get("text", ""))
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
                "text": meta.get("text", ""),
                "score": round(rerank_score, 4),
                "lang": meta.get("lang"),
                "strategy": meta.get("strategy", "passage_native"),
                "chunk_id": meta.get("chunk_id"),
                "query_id": meta.get("query_id"),
            })
        combined.sort(key=lambda c: c["score"], reverse=True)
        return combined

    def _is_english_query(self, query: str) -> bool:
        """Return True if >60% of alphabetic characters in query are ASCII."""
        alpha_count = max(1, sum(1 for c in query if c.isalpha()))
        ascii_alpha = sum(1 for c in query if ord(c) < 128 and c.isalpha())
        return (ascii_alpha / alpha_count) > 0.6

    def _detect_lang(self, text: str):
        """Detect the exact Indic language from Unicode script ranges and lexical markers."""
        # 1. Lexical markers for Devanagari disambiguation
        q_words = set(text.lower().split())
        mr_markers = {"आहे", "नाही", "म्हणजे", "काय", "कसे", "कोणती", "कोणते", "कोणता", "कुठे", "झाले", "केले", "मधील", "मध्ये", "यांचे", "त्यांचे", "आणि", "कशी", "किती", "असावा", "करावे", "कोणत्या"}
        hi_markers = {"है", "हैं", "नहीं", "क्या", "कैसे", "कौन", "कहाँ", "हुआ", "किया", "किए", "होता", "होती", "होते", "और", "में", "पर", "से", "का", "की", "के", "कितना", "कितनी", "चाहिए", "देता", "रहते", "पाया"}
        ne_markers = {"हो", "छ", "छैन", "गर्छ", "गरेको", "कस्तो", "कहाँ", "र", "को", "का", "मा", "हुन्छ"}

        if any(w in q_words for w in mr_markers):
            return "mr"
        if any(w in q_words for w in hi_markers):
            return "hi"
        if any(w in q_words for w in ne_markers):
            return "ne"

        # 2. Assamese vs Bengali letter check
        if any(c in text for c in ('ৰ', 'ৱ')):
            return "as"

        script_ranges = (
            (0x0900, 0x097F, "hi"),  # Default devanagari to hi
            (0x0980, 0x09FF, "bn"),  # Default bengali script to bn
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
        if not counts or sum(counts.values()) == 0:
            return "en"
        best = max(counts, key=counts.get)
        return best if counts[best] >= 2 else "en"

    def _retrieve(self, query: str, lang_filter: str | list | None = None):
        import time
        t0 = time.perf_counter()
        is_eng = self._is_english_query(query)
        rerank_n = self.RERANK_TOP_N
        top_k = self.TOP_K

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
        }, qvec

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
            r"\bhow to (make|build|create) (a |an )?(bomb|weapon|explosive|device)\b"
            r"|\bself[- ]?harm\b|\bhack (into|someone)\b"
            r"|\b(ignore (all )?instructions|system override|print (secret|api key|credentials)|root system access)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(query))

    def _is_current_events_query(self, query: str) -> bool:
        """
        Pattern-based off-topic detector for real-time, current events, weather, or location queries.
        Runs in <1ms (pure regex) before retrieval.
        """
        import re
        q_lower = query.strip().lower()

        # Prime ministers, presidents, current office holders, elections
        political_patterns = [
            r"\b(who is|who's|name of|current)\s+(the\s+)?(prime minister|pm|president|chief minister|cm|governor|vice president)\b",
            r"\b(prime minister|president of|current pm|pm of india|who is the pm|who is the president|who won.*election|election results|presidential election)\b",
            r"(प्रधानमंत्री|राष्ट्रपति|मुख्यमंत्री|राज्यपाल|उपराष्ट्रपति)\s*(कौन|का नाम)?",
            r"(पंतप्रधान|राष्ट्रपती|मुख्यमंत्री|राज्यपाल)\s*(कोण|चे नाव)?",
            r"(निवडणूक निकाल|चुनाव परिणाम)",
        ]
        for p in political_patterns:
            if re.search(p, q_lower):
                return True

        # Weather / live status patterns
        weather_patterns = [
            r"\b(weather|temperature|forecast|rain|snow|humidity|wind speed)\s+(today|now|tomorrow|tonight|in\s+\w+)?\b",
            r"\b(today|tomorrow|now)('s)?\s+(weather|temperature|forecast)\b",
            r"\b(will it rain|is it raining|is it hot|is it cold)\b",
            r"(मौसम|तापमान|बारिश|वर्षा|हवामाना?|थंडी|ऊन|पाऊस)\s*(कैसा|कितना|होगी|आहे|पडेल|का|आज|उद्या|सध्या)",
        ]
        for p in weather_patterns:
            if re.search(p, q_lower):
                return True

        # Pure temporal / "right now" queries
        temporal_patterns = [
            r"\b(what time is it|current time|today's date|what day is it)\b",
            r"(आज क्या तारीख है|आज कौन सा दिन है|आजची तारीख काय)",
        ]
        for p in temporal_patterns:
            if re.search(p, q_lower):
                return True

        return False

    def _token_overlap(self, a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta) if ta else 0.0

    def _scripts_match(self, query: str, answer: str) -> bool:
        """Verify that answer script matches query script."""
        import re
        # Only check alphabetic content, ignoring digits, punctuation, and latin technical units
        q_clean = re.sub(r'[\d\s\W]+', '', query)
        a_clean = re.sub(r'[\d\s\W]+', '', answer)
        if not q_clean or not a_clean:
            return True

        q_lang = self._detect_lang(query)
        a_lang = self._detect_lang(answer)

        # Allow matching within script families (e.g. Hindi/Marathi both Devanagari)
        devanagari_langs = {"hi", "mr", "ne"}
        if q_lang in devanagari_langs and a_lang in devanagari_langs:
            return True
        bengali_langs = {"bn", "as"}
        if q_lang in bengali_langs and a_lang in bengali_langs:
            return True

        return q_lang == a_lang

    def _is_plausible_answer(self, query: str, answer: str, source_text: str = "") -> bool:
        """Plausibility validation: rejects hallucinated values, wrong unit formats, etc."""
        import re
        ans_lower = answer.lower()
        q_lower = query.lower()

        # Reject answers that are purely non-informative phrases
        generic_rejects = [
            "unknown", "not mentioned", "not provided", "no information",
            "पता नहीं", "माहित नाही", "उल्लेख नाही", "उपलब्ध नाही"
        ]
        if any(g == ans_lower.strip() for g in generic_rejects):
            return False

        # Reject answers that are unrealistically large numbers (>100,000 without unit qualifier)
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

    # ── NLI Entailment Check ──────────────────────────────────────────────────

    def _check_entailment(self, premise: str, query: str, answer: str) -> float:
        """
        Check contextual entailment of the extracted answer given the source premise.
        Uses mDeBERTa-v3 multilingual NLI. Capped at 128 max tokens for sub-20ms inference.
        """
        import torch
        if not premise or not answer:
            return 0.0
        hypothesis = f"{query} {answer}".strip()
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        if getattr(self, "nli_device", -1) == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        ent_idx = self.nli_model.config.label2id.get("entailment", 0)
        return float(probs[ent_idx].item())

    # ── Decline helper ────────────────────────────────────────────────────────

    def _decline(self, query, reason, timings, transcript=None, debug_score=None):
        print(f"[GUARDRAIL DECLINE] Query: {query!r} | Reason: {reason!r} | Score: {debug_score} | Timings: {timings}")
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
            "not_entailed":           "The extracted answer could not be verified by contextual entailment.",
            "stt_failed":             "Couldn't transcribe audio -- please retry.",
        }
        return {
            "query": query, "transcript": transcript,
            "answer": msgs.get(reason, "Unable to answer."),
            "sources": [], "confidence": 0.0, "grounded": False,
            "guardrail_triggered": reason, "timings_ms": timings,
            "lang_detected": None,
            "debug_qa_score": debug_score,
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

        # Guardrail 1b — out of scope / current events (regex, <1ms)
        if self._is_current_events_query(query):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "out_of_scope", timings)

        # Retrieval — embed + FAISS + hybrid rerank
        detected_lang = "en" if self._is_english_query(query) else self._detect_lang(query)
        chunks, ret_timings, query_vec_cached = self._retrieve(query, lang_filter=detected_lang)
        timings.update(ret_timings)

        # Guardrail 2 — off-topic (top retrieval score too low)
        top_score = chunks[0]["score"] if chunks else 0.0
        if top_score < self.OFF_TOPIC_THRESHOLD:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "off_topic", timings, debug_score=top_score)

        # Guardrail 3 — retrieval confidence
        if not chunks or top_score < self.MIN_RETRIEVAL_SCORE:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "low_retrieval_confidence", timings, debug_score=top_score)

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
            return self._decline(query, "low_qa_confidence", timings, debug_score=best["score"])

        # Guardrail 5 — Query-answer semantic relevance
        # Fast path: Skip re-embedding if QA score is high or lexical overlap is strong
        if best["score"] < 0.35 and self._token_overlap(query, best["answer"]) < 0.20:
            import numpy as np
            answer_vec = self.embed_model.encode([best["answer"]], normalize_embeddings=True)[0]
            relevance = float(np.dot(query_vec_cached.astype(np.float32), answer_vec.astype(np.float32)))
            if relevance < self.MIN_ANSWER_RELEVANCE:
                timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                return self._decline(query, "low_answer_relevance", timings, debug_score=relevance)

        # Guardrail 6 — Script match
        if not self._scripts_match(query, best["answer"]):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "script_mismatch", timings)

        # Guardrail 7 — Domain plausibility check
        source_text = best.get("source_text", "")
        if not self._is_plausible_answer(query, best["answer"], source_text):
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return self._decline(query, "implausible_answer", timings)

        # Guardrail 8 — NLI Entailment with strict latency budget
        NLI_CHECK_MIN = 0.15
        NLI_CHECK_MAX = 0.35
        TIME_BUDGET_CEILING = 175.0  # ms -- leave 15ms safety margin under 190ms

        elapsed_so_far = (time.perf_counter() - t_start) * 1000  # ms
        if NLI_CHECK_MIN <= best["score"] <= NLI_CHECK_MAX and elapsed_so_far < TIME_BUDGET_CEILING:
            t_nli_start = time.perf_counter()
            entailment_score = self._check_entailment(best.get("source_text", ""), query, best["answer"])
            nli_ms = (time.perf_counter() - t_nli_start) * 1000
            timings["nli_ms"] = round(nli_ms, 2)
            print(f"[NLI CHECK] query={query!r} score={best['score']:.4f} entailment={entailment_score:.4f} nli_ms={nli_ms:.1f}ms")
            if entailment_score < 0.3:
                timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                return self._decline(query, "not_entailed", timings, debug_score=entailment_score)
        elif NLI_CHECK_MIN <= best["score"] <= NLI_CHECK_MAX:
            # Ambiguous score but no time budget left -- log this so we can see how often it happens
            print(f"[NLI SKIPPED - TIME BUDGET] query={query!r} score={best['score']} elapsed={elapsed_so_far:.1f}ms")

        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        # With strict language isolation, answers should match query language.
        # No cross-language attribution prefix needed.
        final_answer = best["answer"]

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
        from fastapi import FastAPI, UploadFile, File, Form, Request
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
        import torch, time, os

        api = FastAPI(title="VoxLore — Multilingual Voice RAG")
        api.add_middleware(
            CORSMiddleware, allow_origins=["*"],
            allow_methods=["*"], allow_headers=["*"]
        )

        @api.get("/", response_class=HTMLResponse)
        @api.get("/index.html", response_class=HTMLResponse)
        def index():
            candidate_paths = [
                "/root/frontend/index.html",
                "frontend/index.html",
                "/frontend/index.html",
                "index.html",
                "/root/index.html"
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        return HTMLResponse(content=f.read())
            return HTMLResponse(content="<h1>VoxLore Live Backend</h1>")

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
                "gpu_vectors_active": getattr(self, "gpu_vectors", None) is not None,
                "gpu_vector_err": getattr(self, "gpu_vector_err", None),
                "metric_type": self.faiss_index.metric_type,
                "test_scores": scores[0].tolist(),
                "test_ids": ids[0].tolist(),
                "first_metadata": first_meta,
                "last_metadata": last_meta,
            }

        @api.get("/debug-qa")
        def debug_qa(query: str = "हिरलूम टमाटर क्या है", context: str = "हिरलूम टमाटर एक पुरानी किस्म है जो खुले परागण से उगाई जाती है।"):
            try:
                result = self._extract_answer(query, context)
                chunks, _, _ = self._retrieve(query)
                chunk_results = []
                for c in chunks[:3]:
                    r = self._extract_answer(query, c.get("text", ""))
                    chunk_results.append({"chunk_lang": c.get("lang"), "chunk_text": c.get("text", "")[:100], "answer": r.get("answer"), "score": r.get("score")})
                return {"direct_test": result, "top_chunks": chunk_results}
            except Exception as e:
                import traceback
                return {"error": str(e), "traceback": traceback.format_exc()}

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
        async def text_query(request: Request):
            q = None
            try:
                form = await request.form()
                q = form.get("query")
            except Exception:
                pass
            if not q:
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        q = body.get("query")
                except Exception:
                    pass
            if not q:
                return {"error": "Missing query parameter"}
            try:
                return self._run_query(q)
            except Exception as e:
                import traceback
                print(f"[ERROR in _run_query] {traceback.format_exc()}")
                return {"error": str(e), "traceback": traceback.format_exc()}

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
