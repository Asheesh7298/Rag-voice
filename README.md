# VoxLore — Multilingual Real-Time Voice RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal%20A100-black.svg)](https://modal.com/)
[![Vector search](https://img.shields.io/badge/Vector%20Search-exact%20GPU%20scan-green.svg)](#-system-architecture)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **VoxLore** is a voice question-answering system: speak or type a question in **Hindi, Marathi, or English** and get an answer **extracted from retrieved passages** (no text generation), searched across **13,020,220 passage vectors** held in GPU memory on an NVIDIA A100.

- **Scale:** 13,020,220 vectors from MS MARCO training passages in Hindi, Marathi and English, using three chunking strategies.
- **Hardware:** NVIDIA A100 on Modal; scales to zero when idle.
- **Server latency:** P50 59 ms, P90 72 ms, max 97 ms (224 questions, measured 2026-11-04 — see [eval/results_phase0.md](eval/results_phase0.md)).
- **Accuracy:** not measured yet. An evaluation on held-out questions is in progress.

---

## 🏛️ System Architecture

```
🎙️ Voice or text question (Hindi / Marathi / English)
  ├── Voice: recorded in the browser (MediaRecorder), transcribed by Sarvam AI (saaras:v3)
  └── Text: POST /query
        ↓
  [Guardrail 1]  Unsafe-input patterns (English)
  [Guardrail 2]  Current-events / real-time patterns (English, Hindi, Marathi)
        ↓
  [Language]     Script and marker-word detection (hi / mr / en)
  [Embedding]    multilingual-e5-base, 768-dim, up to 64 tokens (~14 ms)
        ↓
  [Search]       Exact dot-product scan of 13.02M FP16 vectors in GPU memory (~19 ms)
        ↓
  [Rerank]       Same-language filter; dense score + BM25 + word/trigram overlap (~3 ms)
        ↓
  [Guardrail 3]  Score gates on the reranked score (off-topic, low confidence)
        ↓
  [QA Reader]    xlm-roberta-base-squad2 over up to 6 passages, 256 tokens, spans ≤ 35 tokens (~19 ms)
                 "Tell me about…" questions return the best passage's first sentences instead
        ↓
  [Guardrails 4-7] QA confidence, answer relevance, script match, non-answer filter
        ↓
  🔊 JSON answer + sources + per-stage timings; the browser reads the answer aloud
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), MediaRecorder + Web Audio API (recording, waveform), Speech Synthesis (read-aloud) |
| **Backend** | FastAPI, Uvicorn, PyTorch, NumPy, rank-bm25 |
| **Embeddings** | `intfloat/multilingual-e5-base` (768-dim) |
| **QA Model** | `deepset/xlm-roberta-base-squad2` (extractive, SQuAD 2.0) |
| **Vector Search** | 13,020,220 FP16 vectors in GPU memory, exact search (matrix multiply + top-k) |
| **Speech-to-Text** | Sarvam AI `saaras:v3` |
| **Infrastructure** | Modal (NVIDIA A100, Modal Volume for the index), Vercel for the static page |

---

## 📚 Dataset & Multi-Strategy Indexing

- **Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi), `microsoft/ms_marco` v2.1 (English), training splits.
- **Total Vectors:** 13,020,220.
- **Storage:** 20 GB (18.6 GiB) FP16 vector file + 104 MB byte-offset index into the metadata file.

### Chunking Strategies
| Strategy | Description |
| :--- | :--- |
| `passage_native` | The whole passage |
| `fixed_overlap` | 60-word windows with 15-word overlap |
| `semantic_window` | Groups of up to 4 consecutive sentences |

---

## 🛡️ Guardrails

1. **Unsafe input** — English patterns for weapons, self-harm, hacking and prompt-injection phrases.
2. **Current events** — leaders, elections, weather, date/time questions in English, Hindi and Marathi.
3. **Off-topic gate** — the top reranked score is below a threshold.
4. **Low retrieval confidence** — the top reranked score is below a second threshold.
5. **QA confidence** — the span probability is too low, or the answer is empty or punctuation only.
6. **Answer relevance** — a low-confidence answer that shares few words with the question is re-embedded and compared with it.
7. **Script match** — the answer must be in the same script family as the question.
8. **Non-answer filter** — rejects answers such as "unknown" or "not mentioned".

**Known gaps** (from the [baseline measurement](eval/results_phase0.md)): the off-topic gate did not fire on any of the off-topic test questions, and the unsafe-input filter only covers English.

---

## ⚡ Latency (224 Questions on the Live A100, 2026-11-04)

Server-side timings from each response's `timings_ms`; network time is not included.

| Percentile | Server Total | Embedding | Search (13M) | Rerank | QA Extraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P50** | **59.1 ms** | 14.1 ms | 19.2 ms | 2.7 ms | 19.1 ms |
| **P90** | **72.1 ms** | 15.5 ms | 22.1 ms | 3.5 ms | 23.6 ms |
| **Max** | **97.4 ms** | 17.1 ms | 25.4 ms | 8.1 ms | 60.1 ms |

**First load after idle:** the server scales to zero, so the first visitor waits while it starts and loads the 20 GB vector file: 34.5 s and 43.7 s in two measurements, and about 5½ minutes once when storage was slow.

---

## 🚀 Getting Started

### Run the Voice Web UI Locally
```bash
git clone https://github.com/Asheesh7298/Rag-voice.git
cd Rag-voice
python -m http.server 8000
```
Open [http://localhost:8000](http://localhost:8000) and click the microphone to talk. The page uses the deployed backend.

### Deploy the Backend on Modal
Needs a Modal Volume `voice-rag-index` containing `vectors_fp16.bin`, `metadata.jsonl` and `metadata.offsets`, and a Modal Secret `voice-rag-secrets` with `SARVAM_API_KEY`.
```bash
pip install modal
modal deploy modal_app.py
```

### Reproduce the Measurements
Scripts and results are in [eval/](eval/): `parity_check.py` (capture and compare responses) and `coldstart.py` (first-load wait).

---

## 📄 License
MIT License
