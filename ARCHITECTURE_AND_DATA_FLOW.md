# VoxLore — Architecture, Data Flow & System Design Guide

> Complete architecture diagrams, data flow representations, guardrail pipeline schematics, and benchmark performance analysis for VoxLore — Voice-Enabled Multilingual RAG System.

---

## 1. High-Level Architecture Overview

```mermaid
graph TD
    User([🎙️ User Voice / Text Query]) --> Ingest{Input Mode}
    
    %% Input Routing
    Ingest -->|Voice Audio Stream| STT[Sarvam AI Indic STT API]
    Ingest -->|Text Query| Pipeline[RAG Pipeline Entry]
    STT -->|Transcribed Text| Pipeline

    %% Phase 1: Pre-Retrieval
    subgraph "Phase 1: Pre-Retrieval Guardrails (< 1 ms)"
        Pipeline --> G1[Guardrail 1: Unsafe Input Filter]
        G1 -->|Passed| G1b[Guardrail 2: Out-of-Scope Detector]
        G1 -.->|Violated| Decline1["🛡️ Instant Safe Decline"]
        G1b -.->|Violated| Decline2["🛡️ Out-of-Scope Decline"]
    end

    %% Phase 2: Embedding & Retrieval
    subgraph "Phase 2: Dense Retrieval (~12 ms)"
        G1b -->|Passed| Embed["multilingual-e5-small Embedding (384-dim, NFC Normalized, ~8 ms)"]
        Embed --> DenseSearch["FAISS IndexFlatIP Vector Search (top_k=5, ~0.1 ms)"]
    end

    %% Phase 3: Cross-Encoder Gating
    subgraph "Phase 3: Cross-Encoder Relevance Gate (~15 ms)"
        DenseSearch --> CE["ms-marco-MiniLM-L-6-v2 Cross-Encoder"]
        CE -->|"Score < 5.6"| Decline3["🛡️ Decline: Irrelevant Context"]
        CE -->|"Score >= 5.6"| Merge["Dual-Chunk Context Merge (Top 2, 850 chars)"]
    end

    %% Phase 4: GPU Generation
    subgraph "Phase 4: GPU-Accelerated Generation (~450 ms)"
        Merge --> LLM["Qwen2.5-1.5B-Instruct (FP16 on RTX 4050, 3.2 GB VRAM)"]
        LLM --> PostGuard["Post-Generation Guardrails (Refusal Detection, Intent Validation)"]
    end

    %% Phase 5: Response
    subgraph "Phase 5: Response Delivery"
        PostGuard --> Response["Structured Answer (text, grounded, generation_ms, model)"]
        Response --> TTS["Web Speech Synthesis (Browser Audio)"]
        TTS --> FinalUser(["🔊 User Hears Grounded Answer"])
    end

    style Embed fill:#1e293b,stroke:#818cf8,stroke-width:2px,color:#fff
    style DenseSearch fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff
    style CE fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#fff
    style LLM fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#fff
    style Response fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff
```

---

## 2. End-to-End Request Sequence & Latency Waterfall

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User
    participant Frontend as 🌐 Web UI
    participant E5 as 🧠 E5-Small Embedder
    participant FAISS as 💾 FAISS Vector Index
    participant CE as 🔍 Cross-Encoder Gate
    participant GPU as ⚡ Qwen2.5-1.5B (RTX 4050)

    User->>Frontend: Speaks / Types Query (Hindi / English)
    Frontend->>E5: Encode query to 384-dim vector
    Note over E5: T+0ms → T+8ms (NFC Normalize + Encode)
    E5-->>FAISS: Query vector
    
    FAISS-->>CE: Top-5 candidate passages + scores
    Note over FAISS: T+8ms → T+8.1ms (IndexFlatIP search)
    
    CE->>CE: Score each (query, passage) pair
    Note over CE: T+8.1ms → T+23ms (Cross-Encoder relevance scoring)
    
    alt max_ce < 5.6 (Unanswerable)
        CE-->>Frontend: Decline: grounded=False (~23ms total)
    else max_ce >= 5.6 (Answerable)
        CE->>GPU: Merged context (top 2 chunks, 850 chars)
        Note over GPU: T+23ms → T+473ms (Qwen FP16 generation, 38 tokens)
        GPU-->>Frontend: Answer + grounded=True + generation_ms
    end
    
    Frontend->>User: Renders Answer + Voice Synthesis
```

---

## 3. Multi-Strategy Chunking Pipeline

```mermaid
graph LR
    subgraph "Raw Dataset Sources"
        D1["ai4bharat/MSMARCO-XI (Hindi Passages)"]
        D2["ai4bharat/MSMARCO-XI (English Passages)"]
    end

    subgraph "Multi-Strategy Chunking (3.52x Expansion)"
        D1 & D2 --> S1["Strategy 1: Native Passage (Exact boundaries, 1.00x)"]
        D1 & D2 --> S2["Strategy 2: Fixed Overlap (60 tokens / 15 overlap, 1.41x)"]
        D1 & D2 --> S3["Strategy 3: Semantic Window (Cosine >= 0.55 grouping, 1.11x)"]
    end

    subgraph "Embedding & Indexing"
        S1 & S2 & S3 --> Chunks["Expanded Chunk Corpus"]
        Chunks --> ModelEmbed["multilingual-e5-small Embedding Engine"]
        ModelEmbed --> FaissIdx["FAISS IndexFlatIP (Cosine Similarity)"]
    end

    style S1 fill:#1e40af,stroke:#60a5fa,color:#fff
    style S2 fill:#7c3aed,stroke:#a78bfa,color:#fff
    style S3 fill:#059669,stroke:#34d399,color:#fff
```

---

## 4. RTX 4050 GPU Memory Layout

| Component | Size / VRAM | Compute Speed |
|:---|:---|:---|
| **Qwen2.5-1.5B-Instruct (FP16)** | **3.2 GB** | ~450 ms median, ~1,277 ms P95 generation |
| **Cross-Encoder (MiniLM-L-6-v2)** | **~0.1 GB** | ~15 ms per 5-pair batch |
| **multilingual-e5-small (CPU)** | **0 GB GPU** (runs on CPU) | ~8 ms embedding |
| **Free VRAM Headroom** | **~3.1 GB** | Safety buffer for dynamic batching |
| **Total RTX 4050 VRAM** | **6.44 GB** | Zero OOM risk |

---

## 5. Guardrail Defense Architecture

```mermaid
flowchart TD
    Q[Incoming Query] --> C1{"Guardrail 1: Unsafe / Injection?"}
    C1 -- Yes --> R1["Decline: Unsafe Input (< 0.1ms)"]
    C1 -- No --> C2{"Guardrail 2: Out-of-Scope / Real-Time?"}
    
    C2 -- Yes --> R2["Decline: Out-of-Scope (< 0.1ms)"]
    C2 -- No --> VSearch["Dense Retrieval: E5-Small + FAISS (top_k=5)"]
    
    VSearch --> C3{"Guardrail 3: Cross-Encoder Score >= 5.6?"}
    C3 -- No --> R3["Decline: Irrelevant Context (15ms)"]
    C3 -- Yes --> LLM["Qwen2.5-1.5B Generation (RTX 4050)"]
    
    LLM --> C4{"Guardrail 4: Refusal Pattern Detected?"}
    C4 -- Yes --> R4["Decline: Model Self-Refused"]
    C4 -- No --> C5{"Guardrail 5: Intent Validates? (Address/Phone/Penalty)"}
    
    C5 -- No --> R5["Decline: Intent Mismatch"]
    C5 -- Yes --> Out(["✅ Return Grounded Factual Answer"])

    style Out fill:#166534,stroke:#22c55e,stroke-width:2px,color:#fff
    style R1 fill:#991b1b,stroke:#ef4444,color:#fff
    style R2 fill:#991b1b,stroke:#ef4444,color:#fff
    style R3 fill:#991b1b,stroke:#ef4444,color:#fff
    style R4 fill:#991b1b,stroke:#ef4444,color:#fff
    style R5 fill:#991b1b,stroke:#ef4444,color:#fff
```

---

## 6. Benchmark Performance Summary

### Local RTX 4050 GPU (Eval Loop Results)

| Metric | Result | Target | Status |
|:---|:---|:---|:---|
| **Correctness** | 70.0% – 74.0% | ≥ 70.0% | ✅ PASS |
| **Faithfulness** | 87.0% – 90.0% | ≥ 70.0% | ✅ GOLD STANDARD |
| **Hallucination Rate** | 10.0% – 13.0% | Minimize | ✅ LOW |
| **Recall@1** | 0.560 | Maximize | 56% at Rank 1 |
| **Recall@3** | 0.800 | Maximize | 80% in Top 3 |
| **Recall@5** | 0.920 | Maximize | 92% in Top 5 |
| **MRR** | 0.698 | Maximize | Mean Reciprocal Rank |
| **Retrieval P95** | 11.53 ms | < 50.0 ms | ✅ PASS (77% Headroom) |
| **Generation P95** | 1,277 ms | < 1,500 ms | ✅ PASS (15% Headroom) |
| **False Refusal Rate** | 6.0% | Minimize | ✅ Only 3/50 missed |

### Latency Percentile Breakdown

| Stage | Avg | P50 | P95 | P99 |
|:---|:---|:---|:---|:---|
| **Embed** | 8.07 ms | 7.54 ms | 11.27 ms | 12.45 ms |
| **Search** | 0.12 ms | 0.10 ms | 0.14 ms | 0.18 ms |
| **Retrieval Total** | 8.46 ms | 7.87 ms | 11.53 ms | 12.63 ms |
| **Generation** | 710 ms | 466 ms | 1,277 ms | 1,500 ms |

---

## 7. Model Selection Rationale

We empirically benchmarked 5 model architectures on the RTX 4050 GPU:

| Model | Correctness | Faithfulness | Generation P95 | VRAM | Verdict |
|:---|:---|:---|:---|:---|:---|
| Extractive XLM-RoBERTa | 42.0% | 97.0% | 484 ms | 1.1 GB | Too low correctness |
| Qwen2.5-0.5B-Instruct | 58.0% | 63.0% | 1,304 ms | 1.2 GB | Too many hallucinations |
| **Qwen2.5-1.5B-Instruct (FP16) 🏆** | **74.0%** | **90.0%** | **1,234 ms** | **3.2 GB** | **Champion: Best balanced** |
| Google Gemma-2-2B-It (FP16) | 70.0% | 86.0% | 1,881 ms | 4.9 GB | Over latency budget |
| Qwen2.5-7B-Instruct (4-bit) | 72.0% | 90.0% | 7,500 ms | 5.8 GB | Way too slow |

---

## 8. Video Submission Script (2-Minute Voiceover)

### [0:00 - 0:25] Introduction & Problem Statement
> *"Welcome to VoxLore, a voice-enabled Retrieval-Augmented Generation system built for the HH Goa 2026 Shortlisting Task. Users speak a question in Hindi or English, our pipeline transcribes it using Sarvam AI, retrieves relevant context from the MSMARCO-XI dataset, and returns a grounded factual answer — all within strict latency budgets."*

### [0:25 - 0:55] Architecture & Chunking Strategy
> *"Our architecture combines three key innovations:
> 1. **Multi-Strategy Chunking**: We apply native passage, fixed overlap, and semantic window strategies for 3.52x contextual expansion — not a single naive fixed-size approach.
> 2. **Cross-Encoder Precision Gating**: A dedicated ms-marco-MiniLM relevance filter eliminates irrelevant passages in 15 milliseconds before they reach the LLM.
> 3. **GPU-Accelerated Generation**: Qwen2.5-1.5B running in FP16 on an NVIDIA RTX 4050 generates focused factual answers in under 500 milliseconds median."*

### [0:55 - 1:30] Guardrails & Grounding
> *"Voice assistants cannot afford hallucinations. VoxLore enforces a 5-stage guardrail pipeline:
> - Pre-retrieval filters intercept unsafe and out-of-scope queries in under 0.1 milliseconds.
> - The Cross-Encoder gate rejects irrelevant passages with configurable thresholds.
> - Post-generation validators detect model self-refusals and verify intent-specific grounding.
> The result: 87–90% faithfulness with only 10–13% hallucination rate."*

### [1:30 - 2:00] Benchmarks & Conclusion
> *"On the official rag-local-eval-loop benchmark with 100 queries from MSMARCO-XI:
> - Correctness: 70–74%, passing the competition target.
> - Retrieval P95: 11.5 milliseconds — 77% headroom below the 50ms budget.
> - Generation P95: 1,277 milliseconds — safely within the 1,500ms target.
> - And our Recall@5 is 92%, finding the correct passage 46 out of 50 times.
> Thank you for your time, and we invite you to explore our code and live demo!"*
