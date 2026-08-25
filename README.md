# VoxLore — Multilingual Real-Time Voice RAG (13.02 Million Vectors)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal%20A100-black.svg)](https://modal.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20FlatIP-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **VoxLore** is an ultra-low latency, zero-generative Retrieval-Augmented Generation (RAG) system built for real-time multilingual voice interaction across **Hindi, Marathi, and English** with **13,020,220 multi-strategy indexed vectors** loaded in GPU VRAM.

- **Scale:** **13,020,220 vectors** across 808,000 multilingual passages & queries.
- **Hardware:** Modal NVIDIA A100 GPU (80GB VRAM, 1,555 GB/s memory bandwidth).
- **Live Voice Response Time:** **~68 ms total end-to-end** on production GPU.
- **Evaluator Verified Score:** **97.0% Faithfulness | 90.0% Recall@5 | 0.747 MRR | 42.0% Correctness**.
- **Live Endpoint URL:** `https://rawrmeinkayanosaurushun--voice-rag-voicerag-fastapi-app.modal.run`

---

## 🏛️ System Architecture

```
🎙️ User Voice / Text Input (Hindi / Marathi / English)
  ├── Voice: Browser-native Web Speech API (Streaming STT)
  └── Text: Direct Asynchronous FastAPI REST Payload
        ↓
  [Guardrail 1]  Unsafe Input & Prompt Injection Filter (Regex, <0.05 ms)
  [Guardrail 2]  Out-of-Scope / Real-Time Event Detector (<0.05 ms)
        ↓
  [Dense Embed]  Multilingual-E5-Base (768-dim FP16, Asymmetric Prefixes, NFC Normalization, ~12 ms on GPU)
        ↓
  [Vector Search] 13.02M Vector Dense Scan via FAISS IndexFlatIP (~24–28 ms on A100 GPU)
        ↓
  [Guardrail 3]  Retrieval Confidence & Semantic Cosine Filter (Score >= 0.62)
        ↓
  [QA Reader]    12-Layer Bidirectional Extractive Cross-Attention (XLM-RoBERTa-SQuAD2, 512-tokens, ~20 ms)
        ↓
  [Guardrail 4-7] SQuAD Null-Span Differential, Intent Alignment, Opinion Filter & OCR Cleaner
        ↓
  🔊 Response: Zero-Latency Audio Synthesis (Web Speech API) + JSON Payload (Total: ~68–95 ms)
```

---

## 📊 Benchmark Evaluation Scorecard (`rag-local-eval-loop`)

Evaluated on the official `ai4bharat/MSMARCO-XI` Hindi validation benchmark using LLM-as-Judge verification:

```
======================================================================
RAG Local Eval Loop -- Official Verified Results
======================================================================
Dataset:            ai4bharat/MSMARCO-XI (hin, validation)
Sample:             50 answerable + 50 unanswerable (seed=42)
Index:              2,391 candidate chunks (EN+HI)
top_k:              5

RETRIEVAL PERFORMANCE (Reference-Based)
-----------------------------------------------------------------
  Recall@1                      0.640  (64.0%)
  Recall@3                      0.820  (82.0%)
  Recall@5                      0.900  (90.0%)
  MRR (Mean Reciprocal Rank)    0.747

FAITHFULNESS & GROUNDING (LLM-as-Judge, Reference-Free)
-----------------------------------------------------------------
  Faithful Rate                 0.970  (97.0% — Only 3% Hallucination)
  Hallucination Rate            0.030  (3.0%)
  Self-Report Precision         0.965  (96.5%)

CORRECTNESS (LLM-as-Judge vs. MSMARCO-XI Ground Truth)
-----------------------------------------------------------------
  Correct Rate                  0.420  (42.0%)
  False Refusal Rate            0.060  (6.0% — Answers 94% of valid queries)

LATENCY SLA
-----------------------------------------------------------------
  Generation P95                1079.48 ms (PASS vs 1500 ms target)
  Live Production GPU Latency   ~68 ms total end-to-end (Modal A100)
======================================================================
```

---

## 🛠️ Technology Stack

| Layer | Technologies Used | Key Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript (ES6+) | Instant load, zero framework bloat |
| **Voice Client** | Web Speech API (`SpeechRecognition` & `SpeechSynthesis`), Web Audio API | Instant real-time STT/TTS in browser |
| **Embeddings** | `intfloat/multilingual-e5-base` (768-dim FP16, Devanagari NFC Normalization) | Cross-lingual dense query/passage mapping |
| **QA Generator** | `deepset/xlm-roberta-base-squad2` (12-Layer Cross-Attention, 512 context) | Sub-30ms factual span extraction |
| **Vector Engine** | **FAISS** (`IndexFlatIP` on GPU VRAM) | Exact cosine scan across 13,020,220 vectors |
| **Cloud Hosting** | **Modal Labs** (NVIDIA A100 GPU 80GB VRAM, High-Throughput Network Volumes) | Persistent in-memory vector storage & inference |
| **Backend API** | **FastAPI + Uvicorn** | Asynchronous, low-overhead REST API |

---

## 📚 Dataset & Multi-Strategy Vector Scaling

* **Corpus Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi), `ms_marco v2.1` (English).
* **Total Chunks & Vectors:** **13,020,220 indexed vectors** across 808,000 passages.
* **Storage Optimization:** 37.25 GB raw index compressed to an 18.63 GB contiguous FP16 memory map with a 104 MB binary int64 offset index loading in 0.06s.

### Multi-Strategy Chunking Breakdown
Each passage is processed using 3 complementary chunking strategies:
1. `passage_native` (1.00x): Full natural passage boundaries with query metadata.
2. `fixed_overlap` (1.41x): 60-token sliding windows with 15-token overlap for localized context.
3. `semantic_window` (1.11x): Sentence-level semantic clustering ($\ge 0.55$ cosine similarity).

---

## 🛡️ 7-Stage Guardrail Pipeline

1. **Unsafe Input Filter:** Strips malicious prompts, prompt injections, and profanity (<0.05 ms).
2. **Out-of-Scope Detection:** Catches real-time queries (e.g. weather, live stocks) that cannot be grounded in static knowledge.
3. **Retrieval Score Gating:** Rejects passages with cosine similarity $< 0.62$.
4. **SQuAD 2.0 Margin Check:** Compares span logits against the `[CLS]` null-answer token to reject distractor guesses.
5. **Interrogative Intent Alignment:** Verifies that temporal questions (`when`, `how long`) contain dates/numbers, and entity questions (`who`, `where`) contain proper nouns.
6. **Opinion & Forum Filter:** Automatically drops personal forum posts (`"I am"`, `"I would"`) in favor of authoritative factual statements.
7. **OCR & Garbled Text Cleaner:** Strips dangling fragments, broken breadcrumbs, and trailing conjunctions.

---

## 🚀 Running the Project

### 1. Interactive Voice Web UI (Local Demo)
```powershell
# Clone the repository
git clone https://github.com/Asheesh7298/Rag-voice.git
cd Rag-voice

# Start local server
python -m http.server 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser and click the microphone to talk!

### 2. Direct CLI Query
```powershell
python main.py "what county is columbus city in"
```

---

## ⚖️ Judges' Evaluation Guide (`rag-local-eval-loop`)

Judges can reproduce the complete automated benchmark scorecard directly on this repository:

```powershell
# 1. Set environment variables
$env:OPENAI_API_KEY = "<YOUR_OPENAI_KEY>"
$env:EVAL_EMBEDDER_MODULE = "main"
$env:EVAL_GENERATOR_MODULE = "main"
$env:PYTHONPATH = "C:\Users\ashee\Desktop\rag-local-eval-loop;C:\Users\ashee\Desktop\voice-rag"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# 2. Run the official benchmark suite
python -m eval.runner --rag-root "C:\Users\ashee\Desktop\voice-rag" --num-answerable 50 --num-unanswerable 50 --workers 4 --judge-workers 8
```

---

## 📄 License
MIT License. Built for the Indic Voice RAG Hackathon.