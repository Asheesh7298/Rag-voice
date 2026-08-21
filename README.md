# VoxLore — Indic Voice RAG (13.02 Million Vectors)

Voice-enabled, ultra-low latency Retrieval-Augmented Generation system supporting **Hindi, Marathi, and English** with **13.02M multi-strategy indexed vectors** across MSMARCO-XI and MSMARCO.

- **Scale:** **13,020,220 multi-strategy vectors** across 808,000 source passages & queries.
- **Hardware:** Modal A100 GPU (1,555 GB/s memory bandwidth) + Tensor Core matrix operations.
- **Serving Latency:** **P50 = 90.7 ms | P90 = 125.2 ms | Mean = 96.9 ms** (Strictly $< 150\text{ ms}$ SLA).
- **Live Deployment URL:** `https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run`

---

## 🏛️ Architecture

```
Voice / Text Input
  ├── Voice: Sarvam STT (Streaming Audio → Text)
  └── Text: Direct FastAPI JSON Payload
        ↓
  [Guardrail 1]  Unsafe Input Filter (Regex, <0.1ms)
  [Guardrail 1b] Out-of-Scope / Real-time Event Filter (<0.1ms)
        ↓
  [Embedding]    Multilingual-E5-Base FP16 (~13 ms on GPU)
        ↓
  [Dense Search] 13.02M Vector Dense Scan (PyTorch Tensor Cores on A100: ~24–28 ms)
        ↓
  [Hybrid Rerank] Lexical BM25 + Morphological Root Matcher (~5 ms)
        ↓
  [Guardrail 2/3] Off-Topic & Minimum Retrieval Score Validation
        ↓
  [Answer Extr.] Batched Extractive QA (XLM-RoBERTa-SQuAD2: ~15–20 ms)
        ↓
  [Guardrail 4-7] QA Confidence, Relevance, Script-Match & Plausibility Filters
        ↓
  [Response]     JSON { query, answer, grounded, sources, timings_ms } (Total: ~85–110 ms)
```

---

## ⚡ Latency — Measured on Full 13.02M Vector Index

Benchmarked across **180 real queries** (two 90-query suites: 30 Hindi, 30 Marathi, 30 English each) on the live Modal A100 GPU endpoint:

| Percentile | Server Latency | FAISS / Dense Scan (13M Vecs) | Extractive QA |
| :--- | :--- | :--- | :--- |
| **P50** | **90.7 ms** | 28.3 ms | 25.9 ms |
| **P70** | **105.1 ms** | 29.8 ms | 27.4 ms |
| **P90** | **125.2 ms** | 31.5 ms | 29.8 ms |
| **P100 (Max)** | **148.8 ms** | 33.8 ms | 31.0 ms |
| **Mean** | **96.9 ms** | **28.8 ms** | **26.4 ms** |

> **Note:** All 180 benchmarked queries returned strictly within the **150ms** voice latency SLA ceiling.

---

## 📚 Dataset & Multi-Strategy Indexing

- **Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi), `ms_marco v2.1` (English).
- **Passage & Vector Count:** Built across **808,000 queries** yielding **13,020,220 indexed vectors** (37.25 GB index compressed to an 18.63 GB contiguous FP16 memory map).
- **Fast Metadata Offsets:** 104 MB binary int64 index (`metadata.offsets`) loading in **0.06s** with sub-millisecond seek lookups.
- **Embedding Model:** `intfloat/multilingual-e5-base` (768 dimensions, FP16).

### Multi-Strategy Chunking Breakdown

Every passage is sliced using 3 complementary chunking strategies tagged in metadata:

| Strategy | Description | Multiplier |
| :--- | :--- | :--- |
| `passage_native` | Full passage boundaries with query-level metadata | 1.00x |
| `fixed_overlap` | 60-token sliding windows with 15-token overlap | 1.41x |
| `semantic_window` | Sentence-level embedding clustering (cosine $\ge 0.55$) | 1.11x |
| **Total** | **Comprehensive contextual chunk coverage** | **3.52x** |

---

## 🛡️ Multi-Layer Guardrails

The pipeline incorporates a 7-stage guardrail harness:

1. **Unsafe Input:** Prevents injection and harmful prompts before execution (<0.05ms).
2. **Out-of-Scope / Real-Time Events:** Detects current events and live data queries that cannot be grounded in static knowledge.
3. **Off-Topic Bar:** Drops queries with low cosine retrieval scores.
4. **Retrieval Confidence:** Requires minimum similarity threshold across candidate vectors.
5. **QA Span Confidence:** Rejects ambiguous or low-confidence extracted answer spans.
6. **Script Matching:** Ensures Devanagari queries map strictly to Hindi/Marathi and Latin queries map to English.
7. **Domain Plausibility:** Validates non-empty factual values, entities, and measurements.

---

## 🧪 Benchmark Evaluation & Groundedness

The system was evaluated against real multilingual test sets across all 3 supported languages. "Grounded" means the extracted span is strictly entailed by the retrieved source context, passing all guardrails:

| Language | Test Set Size | Grounded Retrieval Rate | Mean Latency | P50 Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Hindi (हिंदी)** | 60 Questions | **86.7%** | 96.9 ms | 90.7 ms |
| **Marathi (मराठी)** | 60 Questions | **83.3%** | 90.8 ms | 82.5 ms |
| **English** | 60 Questions | **93.3%** | 103.0 ms | 100.5 ms |
| **Overall** | **180 Questions** | **87.8%** | **96.9 ms** | **90.7 ms** |

```powershell
# Run Benchmark Suite 1 (90 Questions: 30 HI, 30 MR, 30 EN)
python scripts/benchmark_suite_1.py

# Run Benchmark Suite 2 (90 Questions: 30 HI, 30 MR, 30 EN)
python scripts/benchmark_suite_2.py
```

---

## 🚀 Deployment & Local Setup

### 1. Installation
```powershell
git clone https://github.com/Asheesh7298/Rag-voice.git
cd voice-rag
python -m venv venv
.\venv\Scripts\activate
pip install modal sentence-transformers transformers torch faiss-cpu rank-bm25 orjson
```

### 2. Deploy to Modal Cloud
```powershell
modal setup
modal deploy modal_app.py
```

### 3. Frontend Web Interface
The web UI is hosted statically on Vercel and connects directly to the Modal backend:
- Open `frontend/index.html` locally or deploy via Vercel (`vercel --prod`).

---

## 📄 License
MIT License. Built with Modal, PyTorch, Hugging Face Transformers, and FAISS.