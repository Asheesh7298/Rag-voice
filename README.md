# VoxLore — Multilingual Real-Time Voice RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal%20A100-black.svg)](https://modal.com/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20%2B%20CuPy-005CED.svg)](https://onnxruntime.ai/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB.svg)](https://react.dev/)

> **VoxLore** is a voice question-answering system: speak or type a question in **Hindi, Marathi or English** and get
> an answer **extracted from retrieved passages** (no text generation), searched across **13,020,220 passage vectors**
> held in GPU memory on an NVIDIA A100.

**🌐 Live demo: [rag-voice-frontend.vercel.app](https://rag-voice-frontend.vercel.app)** — the first question after the
server has been idle takes about 30 s while it starts; after that answers come back in well under a second.

- **Scale:** 13,020,220 vectors from MS MARCO training passages (Hindi 4.60M, Marathi 4.27M, English 4.16M), three chunking strategies.
- **Stack:** React + TypeScript frontend; FastAPI on Modal; models on ONNX Runtime and vector search with CuPy (no PyTorch, no FAISS).
- **Server latency:** P50 46.6 ms, P90 56.3 ms over 627 evaluation questions (2026-11-04).
- **Retrieval:** the exact gold passage is in the top 5 sources for 34% of in-index questions (English 55%); see [Evaluation](#-evaluation).
- **First load after idle:** usually 26–44 s while the server starts and loads 20 GB of vectors (it scales to zero).

---

## 🏛️ System Architecture

```
🎙️ Voice or text question (Hindi / Marathi / English)
  ├── Voice: recorded in the browser (MediaRecorder) → POST /voice-query → converted to 16 kHz WAV (ffmpeg)
  │          → Sarvam AI speech-to-text (saaras:v3)
  └── Text:  POST /query
        ↓
  [Guardrail 1]  Unsafe input: English patterns + a Hindi/Marathi word check
  [Guardrail 2]  Current events / real time: English, Hindi, Marathi patterns
        ↓
  [Language]     Script and marker-word detection → hi / mr / en
  [Embedding]    multilingual-e5-base on ONNX Runtime (GPU), 768-dim, up to 64 tokens        ~4.6 ms
        ↓
  [Search]       Exact FP16 dot-product scan of the question's language block with CuPy      ~15 ms
                 (rows are stored hi | mr | en, so each search covers 4.2–4.6M vectors)
        ↓
  [Rerank]       Dense score + BM25 + word/trigram overlap; duplicate passages removed        ~4 ms
        ↓
  [Guardrail 3]  Retrieval-confidence gate
        ↓
  [QA reader]    xlm-roberta-base-squad2 on ONNX Runtime over up to 6 passages;               ~19 ms
                 answer = best span (≤ 35 tokens) inside a passage
                 "Tell me about…" questions return the best passage's first sentences instead
        ↓
  [Guardrails 4–7] QA confidence, answer relevance, script match, non-answer filter
        ↓
  🔊 JSON answer + sources + per-stage timings; the web app shows them and can read the answer aloud
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite; MediaRecorder + Web Audio API (recording, waveform); Speech Synthesis (read-aloud) |
| **Backend** | FastAPI on Modal, NumPy, rank-bm25, `tokenizers`, ffmpeg (voice recordings → WAV) |
| **Models** | `intfloat/multilingual-e5-base` (official ONNX export) and `deepset/xlm-roberta-base-squad2` (converted to ONNX), run with ONNX Runtime GPU |
| **Vector search** | 13,020,220 FP16 vectors in GPU memory; exact search with a CuPy matrix-vector product and block-wise top-k |
| **Speech-to-text** | Sarvam AI `saaras:v3` |
| **Infrastructure** | Modal (NVIDIA A100, one container at most, scales to zero), Modal Volume for the index; Vercel for the web app |

---

## 📁 Repository Layout

```
modal_app.py      Backend: Modal image, startup, the question pipeline and guardrails, API (/health, /query, /voice-query)
backend/          encoders.py (e5 + QA reader on ONNX Runtime), search.py (CuPy vector search, parallel loading)
frontend/         React + TypeScript web app (Vite); public/presentation.html
scripts/          build_index.py (rebuild the index from the datasets), deploy_modal.py
eval/             Evaluation sets, scripts and the results of every phase of the migration
```

---

## 📚 Dataset & Multi-Strategy Indexing

- **Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi) and `microsoft/ms_marco` v2.1 (English), training splits.
  For each question: its selected passages plus its first passage; exact duplicates removed.
- **Storage on the Modal Volume:** `vectors_fp16.bin` (20 GB, FP16, L2-normalized), `metadata.jsonl`, `metadata.offsets`
  (byte offset of each metadata line), and `onnx/xlm-roberta-base-squad2/` (the converted QA model).
- **Rebuild:** `modal run scripts/build_index.py` (GPU). A test build (`--limit 100`) reproduces the first rows of the live
  index exactly: same chunks in the same order, vectors at cosine 0.99999.

| Strategy | Description |
| :--- | :--- |
| `passage_native` | The whole passage |
| `fixed_overlap` | 60-word windows, starting every 45 words |
| `semantic_window` | Groups of up to 4 consecutive sentences |

---

## 🛡️ Guardrails

1. **Unsafe input** — English patterns (weapons, self-harm, hacking, prompt injection); in Hindi/Marathi, a weapon or explosive word together with a "make" verb, or self-harm phrasing.
2. **Current events** — leaders, elections, weather, date/time questions in English, Hindi and Marathi.
3. **Retrieval confidence** — the best reranked score is below a threshold.
4. **QA confidence** — the span probability is too low, or the answer is empty or punctuation only.
5. **Answer relevance** — a low-confidence answer that shares few words with the question is re-embedded and compared with it.
6. **Script match** — the answer must be in the same script family as the question.
7. **Non-answer filter** — rejects answers such as "unknown" or "not mentioned".

**Off-topic questions** are not reliably caught: a similarity-based gate was calibrated and left off, because off-topic
questions score as high as answerable ones (cosine 0.86–0.92 vs a median of 0.88). Catching them needs another signal,
such as a cross-encoder or a small classifier ([details](eval/results_phase6.md)).

---

## 📊 Evaluation

Measured on the deployed backend with three question sets ([eval/sets/](eval/sets/)): **A**, 300 training questions
whose gold passage is in the index; **B**, 300 validation questions the index never saw; **C**, 27 unsafe,
current-events and off-topic questions. Answers are compared automatically with the dataset's answers.

| | Hindi | Marathi | English | All |
|---|---|---|---|---|
| Set A Recall@5 (gold passage in the top 5 sources) | 27% | 20% | 55% | 34% |
| Set A MRR | 0.19 | 0.11 | 0.40 | 0.23 |
| Set A answer contained in the dataset answer | 13% | 5% | 29% | 16% |
| Set B answered (not refused) | 94% | 91% | 96% | 94% |

Set C: all 11 current-events questions and all 8 unsafe questions (6 English, 2 Hindi/Marathi) declined; 3 of 8
off-topic questions declined.
Dataset answers are often full sentences while this system extracts short spans, so word-overlap scores are low by
construction; the numbers are most useful for comparing versions.

**Latency** (server-side, P50 / P90): total 46.6 / 56.3 ms — embedding 4.6 / 5.3, search 15.4 / 17.9,
rerank 4.0 / 5.0, QA 19.3 / 26.6. The round trip from India adds about 1.2–1.3 s of network time.

The migration from the original PyTorch version is documented phase by phase in [eval/](eval/): baseline
([phase 0](eval/results_phase0.md)), cleanup ([1](eval/results_phase1.md)), evaluation ([2](eval/results_phase2.md)),
ONNX Runtime + CuPy with 224/224 identical answers ([3](eval/results_phase3.md)), startup ([4](eval/results_phase4.md)),
React frontend ([5](eval/results_phase5.md)) and behavior fixes ([6](eval/results_phase6.md)).

---

## 🚀 Getting Started

### Run the Web App Locally
```bash
git clone https://github.com/Asheesh7298/Rag-voice.git
cd Rag-voice/frontend
npm install
cp .env.example .env.local   # VITE_API_URL = the backend the app calls
npm run dev
```

### Deploy the Backend on Modal
Needs the Modal Volume `voice-rag-index` (files listed above) and a Modal Secret `voice-rag-secrets` with at least
`SARVAM_API_KEY` (see [.env.example](.env.example)). The QA model folder on the Volume was converted once from
`deepset/xlm-roberta-base-squad2` (revision `a5fab99`); its `export_info.json` records the versions used and the check
against the original model.
```bash
pip install -r requirements.txt   # the Modal command-line tool
modal deploy modal_app.py
```

### Reproduce the Measurements
```bash
python eval/run_eval.py --url <backend URL> --label <name> --workers 8                    # accuracy + latency
python eval/parity_check.py capture --url <backend URL> --out eval/runs/<name>.json --label <name>
python eval/parity_check.py compare eval/baseline_v1.json eval/runs/<name>.json            # same answers?
python eval/coldstart.py --url <backend URL> --runs 1 --idle 540 --out eval/runs/<name>.jsonl
```

---

## 💸 Running Costs

The backend scales to zero, so it costs nothing while idle. The first visitor after 7 idle minutes starts a container
(A100, about $2.50/hour while running), which stays up for 7 minutes after the last request. At most one container
runs; simultaneous requests queue on it.

---

## 📄 License
MIT License
