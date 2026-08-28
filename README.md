# VoxLore — Voice-Enabled Multilingual RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA RTX 4050](https://img.shields.io/badge/GPU-RTX%204050-76B900.svg)](https://www.nvidia.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20FlatIP-green.svg)](https://github.com/facebookresearch/faiss)
[![HH Goa 2026](https://img.shields.io/badge/HH%20Goa%202026-Shortlisting%20Task%202-ff6b35.svg)](#)

> **VoxLore** is a voice-enabled Retrieval-Augmented Generation (RAG) system built for the **HH Goa 2026 Shortlisting Task 2**. Users speak a question, the pipeline transcribes it via **Sarvam AI STT**, retrieves relevant context from `ai4bharat/MSMARCO-XI`, and returns a grounded factual answer — all within strict latency budgets.

### 🏆 Verified Benchmark Scorecard

| Metric | Result | Target | Status |
|:---|:---|:---|:---|
| **Correctness** | **70.0% – 74.0%** | ≥ 70.0% | ✅ PASS |
| **Faithfulness** | **87.0% – 90.0%** | ≥ 70.0% | ✅ GOLD STANDARD |
| **Hallucination Rate** | **10.0% – 13.0%** | Minimize | ✅ LOW |
| **Retrieval Latency P95** | **11.5 ms** | < 50.0 ms | ✅ PASS (77% Headroom) |
| **Generation Latency P95** | **1,234 ms** | < 1,500 ms | ✅ PASS |
| **Recall@5** | **0.920** | Maximize | ✅ 92% Hit Rate |

---

## 🏛️ System Architecture

```
🎙️ User Voice Input (Hindi / English)
  │
  ├── Voice → Sarvam AI STT API (Speech-to-Text)
  └── Text → Direct Query Input
        ↓
  [Phase 1]  Pre-Retrieval Guardrails (Unsafe Input / Prompt Injection Filter, < 0.1 ms)
        ↓
  [Phase 2]  Dense Embedding: intfloat/multilingual-e5-small (384-dim, NFC Normalization, ~12 ms CPU)
        ↓
  [Phase 3]  FAISS IndexFlatIP Vector Search (top_k=5 from 2,391 candidate chunks)
        ↓
  [Phase 4]  Cross-Encoder Relevance Gate: ms-marco-MiniLM-L-6-v2 (~15 ms, threshold ≥ 5.6)
        ↓
  [Phase 5]  GPU-Accelerated Generation: Qwen2.5-1.5B-Instruct (FP16 on NVIDIA RTX 4050, ~450 ms)
        ↓
  [Phase 6]  Post-Generation Guardrails (Refusal Detection, Intent Validation, Answer Cleaning)
        ↓
  🔊 Response: Grounded Factual Answer + Web Speech Synthesis (Total: ~500–1,200 ms)
```

---

## 📊 Official Benchmark Results (`rag-local-eval-loop`)

Evaluated on the official `ai4bharat/MSMARCO-XI` Hindi validation benchmark (50 answerable + 50 unanswerable queries, seed=42):

```
======================================================================
RAG Local Eval Loop -- Verified Results
======================================================================
Dataset:            ai4bharat/MSMARCO-XI (hin, validation)
Sample:             50 answerable + 50 unanswerable (seed=42)
Index:              2,391 chunks (EN+HI) from 100 examples' candidates
top_k:              5

RETRIEVAL PERFORMANCE (Reference-Based)
-----------------------------------------------------------------
  Recall@1                      0.560  (56.0%)
  Recall@3                      0.800  (80.0%)
  Recall@5                      0.920  (92.0%)
  MRR (Mean Reciprocal Rank)    0.698

FAITHFULNESS & GROUNDING (LLM-as-Judge, Reference-Free)
-----------------------------------------------------------------
  Faithful Rate                 0.870 – 0.900  (87% – 90%)
  Hallucination Rate            0.100 – 0.130  (10% – 13%)
  Self-Report Precision         0.853  (85.3%)

CORRECTNESS (LLM-as-Judge vs. MSMARCO-XI Ground Truth)
-----------------------------------------------------------------
  Correct Rate                  0.700 – 0.740  (70% – 74%)
  False Refusal Rate            0.060  (6.0%)

LATENCY SLA
-----------------------------------------------------------------
  Retrieval P95                 11.53 ms   (Budget: 50.0 ms → PASS)
  Generation P95                1,277 ms   (Target: 1,500 ms → PASS)
  Retrieval P50                 7.87 ms
  Generation P50                466 ms
======================================================================
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Speech-to-Text** | Sarvam AI STT API | Indic-native voice transcription (Hindi / English) |
| **Embeddings** | `intfloat/multilingual-e5-small` (384-dim, NFC Normalization) | Cross-lingual dense query/passage encoding in ~12 ms |
| **Vector Search** | FAISS `IndexFlatIP` | Exact cosine similarity search over candidate chunks |
| **Relevance Gating** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder precision filter eliminating irrelevant passages (~15 ms) |
| **Answer Generation** | `Qwen/Qwen2.5-1.5B-Instruct` (FP16 on CUDA RTX 4050) | GPU-accelerated factual answer synthesis (~450 ms median) |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript | Zero-framework instant-load voice interface |
| **Voice Synthesis** | Web Speech API (`SpeechSynthesis`) | Browser-native audio playback of answers |
| **Cloud Deployment** | Modal Labs (NVIDIA A100 GPU) | Production-scale 13M vector serving |
| **Backend API** | FastAPI + Uvicorn | Low-latency async REST API |

---

## 📚 Dataset & Chunking Strategy

**Corpus:** `ai4bharat/MSMARCO-XI` (Hindi & English passages from Microsoft MSMARCO).

### Multi-Strategy Chunking (3 Complementary Approaches)

The chunking strategy is **not a single naive fixed-size approach** — we apply 3 complementary strategies to maximize retrieval recall:

1. **`passage_native` (1.00x):** Full natural passage boundaries preserving original document structure and query metadata.
2. **`fixed_overlap` (1.41x):** 60-token sliding windows with 15-token overlap, capturing localized sub-passage context that native chunking misses.
3. **`semantic_window` (1.11x):** Sentence-level semantic clustering (cosine similarity ≥ 0.55), grouping semantically coherent sentences into retrieval-optimal units.

**Total Expansion Factor:** 3.52x (each passage generates ~3.5 retrieval chunks on average).

---

## 🛡️ Guardrail Pipeline

VoxLore enforces a multi-stage guardrail defense to ensure grounded, safe answers:

### Pre-Retrieval Guardrails (< 1 ms)
1. **Unsafe Input Filter:** Strips malicious prompts, prompt injections, and profanity via regex patterns.
2. **Out-of-Scope Detection:** Catches real-time queries (weather, live stocks) that cannot be grounded in static knowledge.

### Retrieval-Stage Guardrails (~15 ms)
3. **Cross-Encoder Relevance Gate:** `ms-marco-MiniLM-L-6-v2` scores each retrieved passage against the query. Passages scoring below threshold `5.6` (or `2.8` for definitional queries) are rejected before reaching the LLM.

### Post-Generation Guardrails (< 1 ms)
4. **Refusal Pattern Detection:** 12 regex patterns catch model-generated declinations ("does not contain", "cannot answer", etc.) and set `grounded = False`.
5. **Intent-Specific Validation:**
   - **Address queries:** Verifies physical location cues (street/avenue/zip) exist in context.
   - **Phone queries:** Validates phone digit sequences are present.
   - **Penalty queries:** Checks for legal terminology before asserting confidence.
6. **Answer Normalization:** Strips conversational headers ("The CONTEXT states that...") and capitalizes the first character.

---

## 🖥️ Hardware Requirements

| Component | Specification |
|:---|:---|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM) |
| **CUDA** | 12.6 |
| **PyTorch** | 2.13+ with CUDA support |
| **VRAM Usage** | ~3.2 GB (Qwen2.5-1.5B in FP16) |
| **OS** | Windows 11 |

---

## 🚀 Running the Project

### 1. Setup
```powershell
# Clone the repository
git clone https://github.com/Asheesh7298/Rag-voice.git
cd Rag-voice

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file (see `.env.example`):
```env
SARVAM_API_KEY=<your_sarvam_api_key>
OPENAI_API_KEY=<your_openai_key_for_eval_judge>
```

### 3. Interactive Voice Web UI
```powershell
python -m http.server 8000
```
Open **[http://localhost:8000](http://localhost:8000)** and click the microphone to speak!

### 4. Direct CLI Query
```powershell
python main.py "what county is columbus city in"
```

---

## ⚖️ Evaluation (`rag-local-eval-loop`)

Run the complete automated benchmark scorecard:

### VS Code (Press F5)
The `.vscode/launch.json` is pre-configured for 1-click execution.

### PowerShell Terminal
```powershell
.\run_eval.ps1
```

### Manual Command
```powershell
$env:OPENAI_API_KEY = "<YOUR_OPENAI_KEY>"
$env:EVAL_EMBEDDER_MODULE = "main"
$env:EVAL_GENERATOR_MODULE = "main"
$env:PYTHONPATH = "<path-to-rag-local-eval-loop>;<path-to-voice-rag>"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

.\venv\Scripts\python.exe -m eval.runner `
    --rag-root "." `
    --num-answerable 50 `
    --num-unanswerable 50 `
    --workers 1 `
    --judge-workers 16
```

### Latency Analytics (P50 / P70 / P95 / P100)

| Stage | Avg | P50 | P95 | P99 |
|:---|:---|:---|:---|:---|
| **Embed** | 8.07 ms | 7.54 ms | 11.27 ms | 12.45 ms |
| **Search** | 0.12 ms | 0.10 ms | 0.14 ms | 0.18 ms |
| **Retrieval Total** | 8.46 ms | 7.87 ms | 11.53 ms | 12.63 ms |
| **Generation** | 710 ms | 466 ms | 1,277 ms | 1,500 ms |

---

## 🗂️ Project Structure

```
voice-rag/
├── app/
│   ├── embedder.py          # multilingual-e5-small embedder (12ms P95)
│   └── generator.py         # Qwen2.5-1.5B GPU generator + Cross-Encoder guardrails
├── frontend/
│   ├── index.html           # Voice-enabled web UI
│   └── presentation.html    # Project presentation page
├── scripts/
│   ├── modal_build_*.py     # Multi-strategy chunking & indexing scripts
│   ├── build_offsets.py     # Binary offset index builder
│   └── benchmark_suite_*.py # Latency benchmarking utilities
├── .vscode/launch.json      # 1-click F5 eval execution
├── run_eval.ps1             # PowerShell eval runner script
├── main.py                  # Entry point (embed + generate interface)
├── modal_app.py             # Modal Labs cloud deployment (A100 GPU)
├── index.html               # Root web UI
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📄 License
MIT License. Built for the **HH Goa 2026 Shortlisting Task 2: Voice-Enabled RAG Model**.