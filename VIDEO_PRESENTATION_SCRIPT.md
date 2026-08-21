# 🎙️ Video Presentation & Live Demo Script (1.5 – 2 Minutes)

This script combines the **Architecture Presentation** ([`presentation.html`](file:///c:/Users/ashee/Desktop/voice-rag/presentation.html)) with a **Live Interactive Demo** of the web application ([`index.html`](file:///c:/Users/ashee/Desktop/voice-rag/index.html)).

---

## ⏱️ Video Overview & Timeline

| Timestamp | Screen Action | Focus Topic |
| :--- | :--- | :--- |
| **0:00 – 0:20** | `presentation.html` (Top KPIs) | Introduction & 13.02M Vector Scale |
| **0:20 – 0:40** | `presentation.html` (Architecture) | A100 GPU Tensor-Core Search (24ms) & 7-Stage Guardrails |
| **0:40 – 1:10** | `index.html` (Live Web UI) | **LIVE DEMO**: Voice / Text Query & Sub-100ms Audio Response |
| **1:10 – 1:30** | `presentation.html` (Benchmarks) | 180-Question Benchmark Results (P50 = 90.7ms) & Closing |

---

## 🎬 Word-for-Word Voiceover Script

---

### 📍 [0:00 – 0:20] Part 1: Intro & Scale
**🖥️ Action:** *Start on `presentation.html`. Highlight the 4 KPI cards (13.02M Vectors, 24.1 ms GPU Search, 90.7 ms P50 Latency).*

> **Voiceover:**
> 
> *"Welcome to **VoxLore** — an ultra-low latency, voice-enabled Multilingual RAG system built for **Hindi, Marathi, and English**.*
> 
> *VoxLore indexes the full MSMARCO and MSMARCO-XI datasets, scaling across **13,020,220 multi-strategy vectors** while delivering real-time responses in **under 95 milliseconds**."*

---

### 📍 [0:20 – 0:40] Part 2: Architecture & Guardrails
**🖥️ Action:** *Scroll down to the Section 1 Architecture diagram on `presentation.html`.*

> **Voiceover:**
> 
> *"To achieve real-time speed over 13 million vectors:
> 1. Queries are embedded using **Multilingual-E5-Base** on GPU in 13 milliseconds.
> 2. A direct **PyTorch Tensor-Core matrix scan** on a Modal **A100 GPU** searches all 13.02 million vectors in just **24.1 milliseconds**.
> 3. **XLM-RoBERTa** extracts the exact factual answer, passing our **7-stage guardrail defense** to prevent hallucinations."*

---

### 📍 [0:40 – 1:10] Part 3: 🚀 LIVE PRODUCT DEMO
**🖥️ Action:** *Click the top button **"← Back to Voice Assistant"** to switch to `index.html`.*
1. *Click on a quick example chip (e.g. `[ENGLISH] Columbus County` or `[HINDI] ब्राइटन फोन`).*
2. *Point to the latency pill **`⚡ 83.6 ms Breakdown ↗`**.*
3. *Click **`🔊 Listen`** to hear the instant audio synthesis, then click **`⏹ Stop`**.*

> **Voiceover:**
> 
> *"Let’s see it in action on our live application.*
> 
> *When we submit a query — like asking about Columbus County — VoxLore searches 13 million vectors and extracts the exact answer in just **83.6 milliseconds**!*
> 
> *We can instantly listen to the grounded answer:*
> *(Click `🔊 Listen` → audio plays)*
> 
> *...and pause or toggle audio controls on demand with zero latency."*

---

### 📍 [1:10 – 1:30] Part 4: Benchmark Results & Closing
**🖥️ Action:** *Switch back to `presentation.html` and scroll to Section 4: Live Benchmark Results table.*

> **Voiceover:**
> 
> *"We rigorously validated VoxLore across **180 real benchmark queries** across Hindi, Marathi, and English:
> - Achieving an **87.8% grounded retrieval rate**.
> - With a **P50 latency of 90.7 ms** and worst-case **148.8 ms** — strictly meeting our sub-150ms voice SLA.
> 
> VoxLore delivers massive scale with instant conversational responsiveness. Thank you!"*

---

## 💡 Quick Tips for Seamless Recording:

1. **Browser Tabs**: Have two tabs open in Chrome/Edge:
   - **Tab 1**: [`presentation.html`](file:///c:/Users/ashee/Desktop/voice-rag/presentation.html)
   - **Tab 2**: [`index.html`](file:///c:/Users/ashee/Desktop/voice-rag/index.html)
2. **Smooth Tab Switching**: At the `0:40` mark, click the button in the header or switch tabs with `Ctrl + Tab`.
3. **Audio Check**: Ensure your screen recorder is capturing both your microphone and desktop audio so the assistant's voice synthesis is heard clearly during the demo!
