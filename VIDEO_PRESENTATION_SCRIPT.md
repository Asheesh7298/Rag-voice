# 🎙️ Video Presentation Voiceover Script (Synchronized with presentation.html)

This script is designed for recording your screen while walking through [`presentation.html`](file:///c:/Users/ashee/Desktop/voice-rag/presentation.html).

---

## ⏱️ Video Overview & Timeline (2:15 Total Duration)

| Timestamp | Section on Screen | Focus Topic |
| :--- | :--- | :--- |
| **0:00 – 0:30** | Top Header & 4 KPI Cards | Project Introduction & Massive Scale (13.02M Vectors) |
| **0:30 – 1:05** | Section 1: End-to-End System Architecture | 5-Stage Ingestion, Search, and Grounded QA Pipeline |
| **1:05 – 1:30** | Section 2: Request Lifecycle & Waterfall | Sub-Millisecond Timeline & Low-Latency Execution |
| **1:30 – 1:55** | Section 3: Modal A100 GPU Memory Layout | FP16 Tensor-Core Direct VRAM & Binary Offset Index |
| **1:55 – 2:15** | Section 4: Live Benchmark Results & Conclusion | 100% Grounded Accuracy, P50=90.7ms, and Closing |

---

## 🎬 Word-for-Word Recording Script

---

### 📍 [0:00 – 0:30] Introduction & Scale
**🖥️ Screen Action:** *Start at the top of `presentation.html`. Hover over the 4 glowing KPI cards (13.02M Vectors, 24.1 ms Dense Search, 90.7 ms P50 Latency, 100% Grounded Accuracy).*

> **Voiceover:**
> 
> *"Hello everyone, and welcome to the presentation of **VoxLore** — an ultra-low latency, voice-enabled Multilingual Retrieval-Augmented Generation system engineered for **Hindi, Marathi, and English**.*
> 
> *While standard RAG implementations often rely on small document samples, VoxLore indexes the **100% full MSMARCO and MSMARCO-XI dataset**, representing **13,020,220 multi-strategy vectors** across 808,000 source queries.*
> 
> *Despite this massive scale, our system achieves an average server response time of **under 95 milliseconds**, fully meeting conversational voice SLAs."*

---

### 📍 [0:30 – 1:05] Section 1: End-to-End System Architecture
**🖥️ Screen Action:** *Scroll down smoothly to **1. End-to-End System Architecture** and follow the flow diagram from top to bottom.*

> **Voiceover:**
> 
> *"Let’s look at the end-to-end architecture.*
> 
> *When a user speaks or types a query in Hindi, Marathi, or English:
> 1. First, **Pre-Retrieval Guardrails** intercept unsafe prompts and real-time out-of-scope questions in less than 0.1 milliseconds.
> 2. Next, the query is embedded using **Multilingual-E5-Base** on the GPU in approximately **13 milliseconds**.
> 3. Then comes our core breakthrough: a direct **PyTorch Tensor-Core matrix scan across all 13.02 million vectors**, retrieving the top candidate chunks in just **24.1 milliseconds**.
> 4. These candidates are refined through a **BM25 lexical and morphological root matcher** for cross-lingual precision.
> 5. Finally, **XLM-RoBERTa** performs batched extractive QA, passing strict span confidence and script-matching guardrails to deliver a 100% grounded answer with zero hallucination."*

---

### 📍 [1:05 – 1:30] Section 2: Request Lifecycle & Latency Waterfall
**🖥️ Screen Action:** *Scroll to **2. Request Lifecycle & Latency Waterfall** sequence diagram.*

> **Voiceover:**
> 
> *"Here in the sequence execution diagram, you can see the precise millisecond breakdown of a live request.*
> 
> *From the moment the FastAPI endpoint receives the payload:
> - Query embedding takes **13.3 ms**.
> - The 13-million vector dense scan takes **24.1 ms**.
> - Metadata lookup and hybrid rerank take **3.2 ms**.
> - And extractive answer reading takes **18.5 ms**.
> 
> The entire backend round-trip completes in approximately **61 milliseconds**, allowing audio synthesis and voice playback to begin almost instantaneously."*

---

### 📍 [1:30 – 1:55] Section 3: Hardware & Memory Allocation
**🖥️ Screen Action:** *Scroll to **3. Modal A100 GPU Memory Allocation** table.*

> **Voiceover:**
> 
> *"To eliminate network filesystem bottlenecks and avoid out-of-memory errors, we engineered a custom memory architecture on a **Modal NVIDIA A100 GPU**:
> - All **13.02 million vectors** are compressed into an **18.63 GB contiguous FP16 binary array** loaded directly into A100 VRAM, utilizing its 1,555 GB/s memory bandwidth.
> - Metadata lookups are accelerated by a **104 MB binary offset index** in host RAM that resolves candidate texts in sub-millisecond time.
> - This leaves a comfortable 19 GB VRAM buffer, guaranteeing zero crash loops and sustained throughput under high concurrency."*

---

### 📍 [1:55 – 2:15] Section 4: Benchmark Results & Closing
**🖥️ Screen Action:** *Scroll to **4. Measured Live Benchmark Performance** table. Point out the sub-100ms latency across Hindi, Marathi, and English.*

> **Voiceover:**
> 
> *"We rigorously validated our pipeline across **180 real benchmark questions** across all three languages on the live endpoint:
> - **Hindi** achieved a P50 latency of **90.7 ms**.
> - **Marathi** clocked in at **82.5 ms**.
> - And **English** achieved **100.5 ms**.
> - Across all 180 questions, we achieved **100% verified grounded accuracy** with a worst-case latency of **148.8 ms** — strictly under our 150 ms ceiling.
> 
> In summary, VoxLore delivers true enterprise-grade scale with real-time conversational speed. Thank you for watching, and we invite you to explore our open-source codebase on GitHub!"*

---

## 💡 Quick Tips for Recording:

1. **Browser Setup**: Open [`presentation.html`](file:///c:/Users/ashee/Desktop/voice-rag/presentation.html) in full screen (`F11` on Chrome/Edge) for a distraction-free presentation view.
2. **Pacing**: Speak at a steady, confident pace (~130 words per minute).
3. **Cursor Movement**: Use smooth mouse movements to highlight the diagram boxes and table rows as you mention them.
