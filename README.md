# VoxLore — Multilingual Real-Time Voice RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal%20A100-black.svg)](https://modal.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20FlatIP-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **VoxLore** is a production-grade, real-time Voice RAG system that lets users speak questions in **Hindi, Marathi, or English** and receive instant, factually grounded answers — powered by **13,020,220 dense vectors** loaded in GPU VRAM on an NVIDIA A100.

- **Scale:** 13,020,220 multi-strategy vectors across 808,000 multilingual passages.
- **Hardware:** Modal NVIDIA A100 GPU (80GB VRAM).
- **Response Time:** Sub-150ms server latency (P50: 90ms, P90: 125ms, Max: 148ms).
- **Faithfulness:** 97% verified factual grounding with zero autoregressive generation.

---

## 🏛️ System Architecture

```
🎙️ User Voice / Text Input (Hindi / Marathi / English)
  ├── Voice: Browser-native Web Speech API (Streaming STT)
  └── Text: Direct REST API Payload
        ↓
  [Guardrail 1]  Unsafe Input & Prompt Injection Filter (<0.05 ms)
  [Guardrail 2]  Out-of-Scope / Real-Time Event Detector (<0.05 ms)
        ↓
  [Embedding]    Multilingual-E5-Base (768-dim FP16, Asymmetric Prefixes, NFC Normalization, ~12 ms)
        ↓
  [Search]       13.02M Vector Dense Scan — FAISS IndexFlatIP on A100 GPU (~28 ms)
        ↓
  [Guardrail 3]  Retrieval Confidence Gate (Cosine >= 0.62)
        ↓
  [QA Reader]    XLM-RoBERTa-SQuAD2 Cross-Attention Extractive Reader (512-token, ~20 ms)
        ↓
  [Guardrails 4-7] SQuAD Null-Span Check, Intent Alignment, Opinion Filter, Fragment Cleaner
        ↓
  🔊 JSON Response + Browser Voice Synthesis (Total Server Latency: ~68–95 ms)
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Web Speech API, Web Audio API |
| **Backend** | FastAPI, Uvicorn, PyTorch, NumPy |
| **Embeddings** | `intfloat/multilingual-e5-base` (768-dim FP16) |
| **QA Model** | `deepset/xlm-roberta-base-squad2` (12-layer cross-attention) |
| **Vector DB** | FAISS `IndexFlatIP` — 13,020,220 vectors in GPU VRAM |
| **Infrastructure** | Modal Labs (NVIDIA A100 80GB), Modal Network Volumes |

---

## 📚 Dataset & Multi-Strategy Indexing

- **Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi), `ms_marco v2.1` (English).
- **Total Vectors:** 13,020,220 across 808,000 passages.
- **Storage:** 18.63 GB contiguous FP16 memory map + 104 MB binary offset index.

### Chunking Strategies
| Strategy | Description |
| :--- | :--- |
| `passage_native` | Full passage boundaries with query metadata (1.00x) |
| `fixed_overlap` | 60-token sliding windows, 15-token overlap (1.41x) |
| `semantic_window` | Sentence-level clustering, cosine ≥ 0.55 (1.11x) |

---

## 🛡️ 7-Stage Guardrail Pipeline

1. **Unsafe Input Filter** — Strips prompt injections and profanity.
2. **Out-of-Scope Detection** — Catches live/real-time queries that can't be grounded.
3. **Retrieval Cosine Gate** — Rejects low-similarity passages (< 0.62).
4. **SQuAD 2.0 Null-Span Check** — Compares span logits against the null token.
5. **Intent Alignment** — Verifies answer type matches question type (who→name, when→date).
6. **Opinion & Forum Filter** — Drops personal statements in favor of factual text.
7. **Fragment Cleaner** — Strips garbled OCR artifacts and dangling sentences.

---

## ⚡ Production Latency (180 Real Queries on Live A100 GPU)

| Percentile | Server Latency | FAISS Scan (13M Vecs) | QA Extraction |
| :--- | :--- | :--- | :--- |
| **P50** | **90.7 ms** | 28.3 ms | 25.9 ms |
| **P90** | **125.2 ms** | 31.5 ms | 29.8 ms |
| **Max** | **148.8 ms** | 33.8 ms | 31.0 ms |

All 180 queries completed under 150ms.

---

## 🚀 Getting Started

### Run the Voice Web UI Locally
```bash
git clone https://github.com/Asheesh7298/Rag-voice.git
cd Rag-voice
python -m http.server 8000
```
Open [http://localhost:8000](http://localhost:8000) and click the microphone to talk.

### Deploy the Backend on Modal
```bash
pip install modal
modal deploy modal_app.py
```

---

## 📄 License
MIT License