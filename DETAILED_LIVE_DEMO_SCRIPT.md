# 🎬 Detailed Live Demo Walkthrough & Voice Script

This guide provides a step-by-step, action-by-action blueprint for demonstrating **VoxLore** during your video recording.

---

## 🎯 Demo Structure & Key Capabilities to Showcase:

1. **Lightning-Fast English Factoid Query** (Sub-90ms latency & exact entity extraction).
2. **Detailed Latency Telemetry Breakdown** (Inspecting the Modal A100 GPU waterfall).
3. **Indic Multilingual Query (Hindi / Marathi)** (Zero translation lag & native Devanagari script).
4. **Interactive Audio Speech & Stop Controls** (`🔊 Listen` $\rightarrow$ `⏹ Stop`).
5. **Guardrail Safety Refusal** (Demonstrating reliable refusal on unanswerable/off-topic inputs).

---

## 📋 Step-by-Step Demo Execution Guide

---

### Step 1: English Query & Instant Extractive QA (~15 seconds)

**🖥️ Screen Action:**
1. On [`index.html`](file:///c:/Users/ashee/Desktop/voice-rag/index.html), click the quick chip:  
   **`[ENGLISH] Columbus County`** *(or type `"what county is columbus city in"`)*.
2. The answer card renders instantly with **`🤖 EXTRACTIVE QA`** and **`Franklin County`**.
3. Point your mouse to the green badge: **`⚡ 83.6 ms`**.

> **🎙️ What to say:**
> 
> *"Let's test an English factoid query: 'what county is columbus city in'.*  
> *In just **83.6 milliseconds**, VoxLore scans all 13.02 million vectors and extracts the exact factual entity: **Franklin County**."*

---

### Step 2: Open Stage-by-Stage Latency Telemetry (~15 seconds)

**🖥️ Screen Action:**
1. Click the button: **`Breakdown ↗`** next to the latency pill.
2. The Telemetry Drawer pops up displaying the full GPU waterfall.
3. Hover your cursor over the progress bars:
   - *Query Embedding (Multilingual-E5): ~13 ms*
   - *Dense Vector Search (13.02M Vectors): ~24 ms*
   - *Hybrid Reranking (BM25): ~3 ms*
   - *Extractive QA Reader (XLM-RoBERTa): ~18 ms*
4. Click the **`✕ Close`** button.

> **🎙️ What to say:**
> 
> *"If we open the latency breakdown, we can see the exact millisecond trace on the Modal A100 GPU:*  
> *Query embedding took 13 milliseconds, the 13-million vector dense scan completed in 24 milliseconds, and XLM-RoBERTa extracted the answer in 18 milliseconds — passing our sub-150ms voice SLA with room to spare."*

---

### Step 3: Indic Query (Hindi / Marathi) & Audio Synthesis (~20 seconds)

**🖥️ Screen Action:**
1. Click the Hindi quick chip: **`[HINDI] मेन अमीर व्यक्ति`** *(or type `"मेन में सबसे अमीर व्यक्ति कौन है"`)*.
2. The answer card renders: **`सुसान अल्फोंड`** in **~92 ms**.
3. Click the button: **`📚 3 Context Sources ▾`** to show the retrieved passages.
4. Click **`🔊 Listen`** — the browser native audio plays the Hindi answer.
5. Click **`⏹ Stop`** (or the red floating stop button) to demonstrate instant audio control.

> **🎙️ What to say:**
> 
> *"Now let's test Hindi: 'मेन में सबसे अमीर व्यक्ति कौन है' (Who is the richest person in Maine?).*  
> *VoxLore retrieves the Indic passage and extracts **सुसान अल्फोंड** (Susan Alfond) in **92 milliseconds**.*  
> *We can inspect the exact context passages from MSMARCO-XI, and click **Listen** for instant voice playback:*  
> *(Click `🔊 Listen` → audio speaks → click `⏹ Stop`)*  
> *...with one-click audio toggle controls."*

---

### Step 4: 7-Stage Guardrail Defense Demonstration (~15 seconds)

**🖥️ Screen Action:**
1. Type an ungrounded or current-events query in the search box:  
   `"who won yesterday's football match"` *(or click an off-topic test)*.
2. Press `Enter`.
3. The answer card immediately appears with the badge **`🛡️ GUARDRAIL`**:  
   *"This query asks about real-time or current events that cannot be verified from our static knowledge base."* in **< 1 ms**.

> **🎙️ What to say:**
> 
> *"Reliability is critical for voice assistants. When given a real-time question like 'who won yesterday's football match', our **Pre-Retrieval Guardrail** intercepts it in under **0.1 milliseconds**, safely declining rather than hallucinating a fake answer."*

---

## 🎯 Summary of Tested Demo Questions:

| Language | Query | Extracted Answer | Response Time |
| :--- | :--- | :--- | :--- |
| **English** | `what county is columbus city in` | **Franklin County** | **83.6 ms** |
| **English** | `how much does it cost to change a jeep alternator` | **$300 to $500** | **88.2 ms** |
| **Hindi** | `मेन में सबसे अमीर व्यक्ति कौन है` | **सुसान अल्फोंड** | **92.4 ms** |
| **Hindi** | `ब्राइटन टाउनशिप फोन नंबर` | **412-774-4800** | **91.1 ms** |
| **Marathi** | `फ्रान्सचे सध्याचे चलन काय आहे` | **युरो (Euro)** | **85.0 ms** |
| **Guardrail** | `who won yesterday's match` | **🛡️ Declined (Real-Time Event)** | **< 0.1 ms** |

---

## 💡 Recording Checklist:
- [ ] **Tab 1 Ready**: `presentation.html`
- [ ] **Tab 2 Ready**: `index.html`
- [ ] **Desktop Audio Enabled**: Make sure your recording software records both microphone voice and speaker audio so the assistant's voice is heard during Step 3.
