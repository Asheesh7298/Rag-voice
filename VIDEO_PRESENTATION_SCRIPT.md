# 🎙️ 90-Second Video Presentation Script (Fast & Punchy)

Designed for recording your screen walkthrough of [`presentation.html`](file:///c:/Users/ashee/Desktop/voice-rag/presentation.html) in exactly **1.5 minutes (90 seconds)**.

---

## ⏱️ Timeline & Screen Actions (90 Seconds Total)

| Timestamp | Screen Section | Topic |
| :--- | :--- | :--- |
| **0:00 – 0:20** | Top Header & 4 KPI Cards | Introduction & Massive 13.02M Vector Scale |
| **0:20 – 0:45** | Section 1 & 2: Architecture & Sequence | A100 GPU Tensor-Core Search (24ms) & Extractive QA |
| **0:45 – 1:10** | Section 3: VRAM Memory Layout | 7-Stage Guardrails & Zero Hallucination |
| **1:10 – 1:30** | Section 4: Live Benchmark Results | 180 Questions Evaluated (P50 = 90.7ms) & Closing |

---

## 🎬 Word-for-Word Voiceover Script (~195 words)

### 📍 [0:00 – 0:20] Intro & Scale
**🖥️ Action:** *Start at the top of `presentation.html`. Highlight the 4 KPI cards.*

> *"Welcome to **VoxLore** — an ultra-low latency, voice-enabled Multilingual RAG system for **Hindi, Marathi, and English**.*
> 
> *VoxLore indexes the full MSMARCO and MSMARCO-XI datasets, scaling across **13,020,220 multi-strategy vectors** while delivering real-time responses in **under 95 milliseconds**."*

---

### 📍 [0:20 – 0:45] Architecture & A100 Acceleration
**🖥️ Action:** *Scroll down to the Section 1 Architecture & Section 2 Sequence diagrams.*

> *"To achieve conversational speed over 13 million vectors:
> 1. Queries are embedded using **Multilingual-E5-Base** on GPU in 13 milliseconds.
> 2. A direct **PyTorch Tensor-Core matrix scan** on a Modal **A100 GPU** searches all 13.02 million vectors in just **24.1 milliseconds**.
> 3. Candidates are refined with **BM25**, and **XLM-RoBERTa** extracts the exact factual answer."*

---

### 📍 [0:45 – 1:10] Memory Layout & Guardrails
**🖥️ Action:** *Scroll to Section 3: Hardware Memory Layout table.*

> *"Our entire vector index is stored as an **18.63 GB FP16 binary array** in A100 VRAM, backed by a **104 MB binary offset index** in host RAM for sub-millisecond lookups.
> 
> To guarantee reliability, our **7-stage guardrail defense** intercepts unsafe queries and strictly declines unanswerable questions to eliminate hallucinations."*

---

### 📍 [1:10 – 1:30] Benchmark Results & Closing
**🖥️ Action:** *Scroll to Section 4: Live Benchmark Results table.*

> *"Across **180 live benchmark queries** spanning Hindi, Marathi, and English:
> - Our system achieved an **87.8% grounded retrieval rate**.
> - With a **P50 latency of 90.7 ms** and worst-case **148.8 ms** — strictly meeting our sub-150ms voice SLA.
> 
> VoxLore delivers massive scale with instant conversational responsiveness. Thank you!"*
