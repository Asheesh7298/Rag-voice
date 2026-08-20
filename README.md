# Voice RAG — Multilingual Indic MSMARCO (1.51M Multi-Strategy Index)

Voice-enabled Multilingual RAG pipeline supporting **Hindi, Marathi, and English** with **sub-150ms retrieval + extractive QA latency**, multi-strategy indexing across **1,516,928 vectors**, and a comprehensive **7-stage zero-hallucination guardrail suite**.

---

## 🏆 Headline Benchmark Evidence (90-Question Evaluation)

Evaluated against the live deployed system on Modal (A10G GPU + 8 vCPUs) across 30 Hindi, 30 Marathi, and 30 English queries.

| Metric | Budget Target | Hindi (30 Qs) | Marathi (30 Qs) | English (30 Qs) | **Overall 90-Question Benchmark** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Grounded Right Answers** | > 85% | **28 / 30 (93.3%)** | **25 / 30 (83.3%)** | **30 / 30 (100.0%)** | **83 / 90 (92.2%)** |
| **Server Latency P50** | < 150 ms | **144.3 ms** | **145.3 ms** | **140.8 ms** | **143.2 ms** |
| **Server Latency P70** | < 160 ms | **147.2 ms** | **150.2 ms** | **143.4 ms** | **147.0 ms** |
| **Server Latency P90** | < 180 ms | **149.5 ms** | **153.9 ms** | **147.8 ms** | **151.1 ms** |
| **Server Latency P100 (Max)** | < 200 ms | **157.6 ms** | **160.8 ms** | **152.1 ms** | **160.8 ms** |
| **Mean Server Latency** | < 150 ms | **144.4 ms** | **145.4 ms** | **141.3 ms** | **143.7 ms** |
| **FAISS Search P50** | < 100 ms | **100.0 ms** | **99.6 ms** | **98.6 ms** | **99.5 ms** |
| **QA Extraction P50** | < 25 ms | **17.7 ms** | **20.9 ms** | **16.4 ms** | **18.0 ms** |

*Official JSON evidence saved in: [`data/benchmark_90_results.json`](data/benchmark_90_results.json) and [`benchmarks/final_results.md`](benchmarks/final_results.md).*

---

## 🧩 1. Multi-Strategy Chunking Pipeline

To maximize retrieval coverage across diverse query formulations, the corpus is chunked using **three distinct, complementary strategies**:

1. **`passage_native`**: Complete, unmodified passage chunks that preserve full document-level context and core factual statements.
2. **`fixed_overlap`**: 60-token sliding windows with 15-token overlap using script-agnostic whitespace tokenization across Hindi, Marathi, and English.
3. **`semantic_window`**: Punctuation- and sentence-boundary-aware chunks (splitting on Devanagari danda `।` and Latin `.!?`, with 2-sentence windows).

### Measured Multiplier & Scale:
- **Measured Chunks-per-Passage Ratio**: **3.5609x** (empirically measured on a 5,000-passage multi-lingual sample).
- **Source Passages**: 426,000 (142,000 HI + 142,000 MR + 142,000 EN).
- **Total Vectors in FAISS Index**: **1,516,928** (4.34 GB FP16 embedding matrix).
  - `passage_native`: 426,000 vectors
  - `fixed_overlap`: 600,523 vectors
  - `semantic_window`: 490,405 vectors

### Chunk Metadata Schema:
Every vector chunk in `metadata.jsonl` tracks full lineage:
```json
{
  "chunk_id": "hi-436492-p1-fixed-w0",
  "source_passage_id": "hi-436492-p1",
  "query_id": "hi-436492",
  "query": "सबसे बड़ा उड़ने वाला सरीसृप अब तक",
  "ground_truth_answer": "प्रोजेक्टोरिया का अज़हद्रिचिडे परिवार",
  "chunk_strategy": "fixed_overlap",
  "lang": "hi",
  "text": "सबसे बड़े उड़ने वाले जानवर जो कभी भी जीवित रहे थे...",
  "window_start_tok": 0,
  "window_size": 60,
  "is_selected": true
}
```

---

## ⏱️ 2. Latency Methodology & Pipeline Breakdown

### Scoping Rule:
The **<200ms latency budget** strictly scopes to the **server-side Retrieval + QA Extraction pipeline** under our direct computational control. External network calls (such as Sarvam STT) are instrumented and reported separately.

```
[User Audio / Mic] ──> Sarvam STT (Network: ~400-800ms) ──> [Text Query]
                                                                  │
   ┌──────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────┐
   │ CORE SERVER PIPELINE (Target: <200ms | Measured: 134-143ms P50)                                                  │
   │                                                                                                                  │
   │ 1. Vector Embed (multilingual-e5-base on A10G GPU) : ~11.1 ms                                                    │
   │ 2. FAISS HNSW Vector Search (8 vCPUs, efSearch=24) : ~98.5 ms                                                    │
   │ 3. Hybrid BM25 & Lang Isolation Boost              : ~2.5 ms                                                     │
   │ 4. Extractive QA (xlm-roberta-base-squad2 on GPU)  : ~17.5 ms (max_length=256)                                   │
   │ 5. Guardrail Validations                           : ~1.2 ms                                                     │
   └──────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                                  │
                                                       [Grounded JSON Answer]
```

---

## 🛡️ 3. 7-Stage Zero-Hallucination Guardrail Suite

The system implements 7 defensive guardrails to ensure it **knows when not to answer**:

| # | Guardrail Name | Threshold / Mechanism | What It Catches & Handles |
| :-: | :--- | :--- | :--- |
| **G1** | **Off-Topic / Gibberish Gate** | `OFF_TOPIC_THRESHOLD = 0.70` | Catches random keyboard mashing, unaligned noise, and out-of-distribution queries before retrieval. |
| **G2** | **Out-of-Scope / Real-Time Events** | Regex + Temporal Event Filter | Catches queries requiring real-time knowledge (e.g. 2024 elections, current political heads, live weather). |
| **G3** | **Unsafe & Injection Filter** | Pattern Matching + Security Rules | Catches prompt injections, jailbreaks, system bypass attempts, and harmful instructions. |
| **G4** | **Retrieval Confidence Gate** | `MIN_RETRIEVAL_SCORE = 0.65` | Declines queries when top retrieved vector similarity is too low to support a factual answer. |
| **G5** | **Answer Relevance Verification** | `MIN_ANSWER_RELEVANCE = 0.20` | Verifies extracted answer is semantically related to the user's question, eliminating topical drift. |
| **G6** | **Extractive QA Score Gate** | `MIN_QA_SCORE = 0.05` | Rejects low-confidence span extractions where XLM-RoBERTa start/end logits indicate ambiguity. |
| **G7** | **Grounding & Overlap Gate** | Lexical & Substring Matching | Ensures extracted answer text exists verbatim within retrieved passages to prevent hallucination. |

---

## 📁 Repository Layout

```
├── modal_app.py                      # Production Modal deployment (A10G GPU, 8 vCPUs, 16GB RAM)
├── frontend/
│   └── index.html                    # Real-time Voice & Text UI with live telemetry & waveform
├── data/
│   ├── benchmark_90_verified.json    # Verified 90-question benchmark dataset (Set 1)
│   ├── benchmark_90_set2_verified.json # Verified 90-question benchmark dataset (Set 2)
│   └── benchmark_90_results.json     # Official latency & accuracy run logs
├── benchmarks/
│   ├── final_results.md              # Full benchmark analysis & evidence documentation
│   └── modal_bench.py                # Automated load testing script
├── scripts/
│   ├── benchmark_90.py               # Official Set 1 benchmark runner
│   ├── benchmark_90_set2.py          # Official Set 2 benchmark runner
│   ├── modal_build_multi_strategy_1_5m.py # Multi-strategy index builder on Modal
│   └── measure_multi_strategy_ratio.py # Empirical chunking multiplier calculation
```

---

## 🚀 Running the System

### 1. Local Web Interface
```powershell
# Start local frontend server on port 8000
python -m http.server 8000 --directory frontend
# Open http://localhost:8000 in your browser
```

### 2. Run Benchmarks Against Live Endpoint
```powershell
# Run Set 1 (90 Questions: 30 HI, 30 MR, 30 EN)
python scripts/benchmark_90.py

# Run Set 2 (90 New Disjoint Questions)
python scripts/benchmark_90_set2.py
```

### 3. Deploy to Modal
```powershell
python -m modal deploy modal_app.py
```
