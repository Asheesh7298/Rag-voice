# Voice RAG — Indic MSMARCO

Voice-enabled Retrieval-Augmented Generation system supporting Hindi, Marathi, and English. Speech → Sarvam STT → multi-strategy chunked retrieval (FAISS) → extractive QA with entailment-verified grounding → answer, with full latency instrumentation and multi-layer guardrails.

**Live endpoint:** `https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run`

## Architecture

```
Voice/Text input
  → Sarvam STT (voice only)                              [external network call, reported separately]
  → Guardrail 1: unsafe input (regex, ~0ms)
  → Guardrail 1b: out-of-scope / current-events detector  (~0ms)
  → Query embedding (multilingual-e5-base, GPU)
  → FAISS HNSW search (1.5M vectors, language-filtered)
  → Hybrid BM25 + dense rerank
  → Guardrail 2: off-topic (retrieval score threshold)
  → Guardrail 3: low retrieval confidence
  → Batched extractive QA (xlm-roberta-base-squad2, GPU)
  → Guardrail 4: low QA confidence
  → Guardrail 5: query-answer semantic relevance (embedding cosine sim)
  → Guardrail 6: script match (answer language == query language)
  → Guardrail 7: domain plausibility (implausible numeric answers)
  → Guardrail 8: NLI entailment check (mDeBERTa, ambiguous-confidence only, time-budgeted)
  → Response {answer, sources, confidence, grounded, timings_ms}
```

## Latency — measured, not estimated

**Methodology:** measured across **180 real test queries** (two independent 90-question sets, 30 Hindi + 30 Marathi + 30 English each, disjoint questions), against the live production endpoint. Reported latency is the full retrieval + extraction + verification pipeline (embed → FAISS search → rerank → QA → guardrails), **excluding STT**, since STT is an external network call to a third-party API and is reported separately.

| Percentile | Latency      |
| ---------- | ------------ |
| P50        | 143.5 ms     |
| P70        | 152.0 ms     |
| P90        | 162.5 ms     |
| **P100**   | **183.1 ms** |

All 180 queries completed under the 200ms target, including worst-case.

STT (Sarvam, network call): typically 500–800ms round trip, reported separately per the same reasoning applied industry-wide — no RAG system can make a third-party network call complete in under 200ms, so the <200ms budget is scoped to the retrieval/generation pipeline under our control.

## Dataset & indexing

- **Sources:** `ai4bharat/MSMARCO-XI` (Hindi, Marathi), Microsoft `ms_marco v2.1` (English)
- **Scale:** ~426,000 source passages (142,000 per language), producing **~1,500,000 indexed vectors** after multi-strategy chunking
- **Embedding model:** `intfloat/multilingual-e5-base`, 768-dim, FP16
- **Index:** FAISS HNSW, quantized/RAM-resident for fast serving, no network-volume mmap on the query path
- **Infra:** Modal, A10G GPU, `min_containers=1`

### Chunking strategy — measured, not assumed

Rather than a single naive fixed-size chunker, every source passage is processed through **three complementary chunking strategies**, each tagged with `chunk_strategy` metadata for retrieval-time filtering and ablation:

| Strategy          | Description                                                                                                                                                      | Measured chunks/passage |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `passage_native`  | One chunk per passage, exact boundaries. Metadata-aware (`source_passage_id`, `query_id`, `is_selected`, `lang`).                                                | 1.00x                   |
| `fixed_overlap`   | 60-token sliding windows, 15-token overlap. Script-agnostic (whitespace tokenization works identically across Hindi/Marathi/English).                            | 1.41x                   |
| `semantic_window` | Sentence-level, embedding-similarity-based grouping (cosine ≥0.55 breakpoint, max 6 sentences/chunk). Handles both Latin punctuation and Devanagari danda (`।`). | 1.11x                   |

Combined measured multiplier: **3.52x chunks per source passage** — measured on a 5,000-passage sample, not assumed, before calculating the final per-language passage allocation needed to hit the target index scale.

## Guardrails — knowing when not to answer

Eight independent guardrail layers, each with a distinct failure mode it catches:

1. **Unsafe input** — regex pattern match (bomb-making, hacking, prompt injection), declines in <0.05ms before any retrieval
2. **Out-of-scope / current events** — weather, real-time data, "who is the current PM", election results — declines before retrieval since these can never be grounded in a static corpus
3. **Off-topic** — top retrieval score below threshold → query has no good match in the corpus
4. **Low retrieval confidence** — no chunk cleared the minimum similarity bar
5. **Low QA confidence** — the extractive QA model itself wasn't confident in any span
6. **Low answer relevance** — query-answer embedding similarity check, catches confidently-extracted-but-irrelevant spans
7. **Script mismatch** — answer language doesn't match query language (e.g. Hindi query, Bengali-script answer)
8. **Domain plausibility** — implausible numeric answers for cost/price queries (e.g. "$800,000 per pitch" for a per-square-foot tile question)
9. **NLI entailment (ambiguous-confidence only)** — for extractions with QA confidence in the 0.15–0.35 range, a lightweight multilingual NLI model (`mDeBERTa-v3-base-xnli`) verifies the retrieved passage actually entails the specific answer given. Time-budgeted (only runs if <175ms elapsed) so it can never push latency past the 200ms ceiling — 0/180 test queries were skipped due to budget in our benchmark, confirming the budget is generous under normal load.

All declines are logged with their triggering reason and the underlying score, for reproducible tuning.

### On accuracy reporting — groundedness vs. correctness

We report **87.8% grounded-answer rate** (158/180) across the two 90-question benchmarks. "Grounded" means the system extracted an answer from retrieved context that passed all 8 guardrail layers, including the entailment check. We are explicit that this measures verified groundedness, not independently fact-checked correctness in every case — extractive QA on Hindi/Marathi with a cross-lingually-transferred model (not natively trained on Indic QA pairs) has a known accuracy ceiling below English (which scored 93.3–100% across our test sets vs. 76.7–93.3% for Hindi/Marathi). Adding the NLI entailment layer measurably improved this: it caught and correctly refused 6+ cases of confidently-wrong extractions (e.g., a query asking about a medical procedure returning an unrelated flower name) that a naive groundedness check alone would have missed.

## Harness

The pipeline is a structured state machine (`modal_app.py`, `VoiceRAG` class), not a single prompt-in/text-out call:

- Explicit per-stage timing instrumentation on every request
- Retry logic on STT network calls (`tenacity`)
- Structured JSON I/O at every stage
- Graceful decline paths with typed reason codes, not raw exceptions
- Time-budgeted conditional guardrail execution (NLI check) to guarantee latency SLA compliance under all conditions

## Setup

```bash
git clone <repo-url> && cd voice-rag
pip install modal
modal setup
modal secret create voice-rag-secrets SARVAM_API_KEY=<key> SARVAM_STT_URL=https://api.sarvam.ai/speech-to-text OFF_TOPIC_THRESHOLD=0.70 MIN_RETRIEVAL_SCORE=0.65 MIN_QA_SCORE=0.05 MIN_ANSWER_RELEVANCE=0.20 TOP_K=5 RERANK_TOP_N=20
modal deploy modal_app.py
```

Benchmark against the live endpoint:

```bash
python scripts/benchmark_90.py
python scripts/benchmark_90_set2.py
```

## API

Base URL: `https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run`

| Method | Route          | Purpose                                                |
| ------ | -------------- | ------------------------------------------------------ |
| `GET`  | `/`            | Serves the web UI                                      |
| `GET`  | `/health`      | Status, index size, loaded models, supported languages |
| `POST` | `/query`       | Text question → grounded answer                        |
| `POST` | `/voice-query` | Audio file → transcript + grounded answer              |
| `GET`  | `/debug-index` | Index diagnostics                                      |
| `GET`  | `/debug-qa`    | Run extractive QA against a supplied query + context   |

```bash
BASE=https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run
```

### GET /health

```bash
curl $BASE/health
```

Returns index size, GPU availability, the loaded QA and embedding models, and the
list of supported languages.

### POST /query

Accepts either a form field or a JSON body:

```bash
curl -X POST $BASE/query -F "query=हिरलूम टमाटर क्या है"

curl -X POST $BASE/query \
     -H "Content-Type: application/json" \
     -d '{"query": "what is an heirloom tomato"}'
```

### POST /voice-query

Multipart upload. `language_code` is optional — omit it to auto-detect.

```bash
curl -X POST $BASE/voice-query \
     -F "file=@question.wav" \
     -F "language_code=hi"
```

The response is identical to `/query`, plus the STT `transcript` and an
additional `stt_ms` entry in `timings_ms`.

### Response

`sources` contains up to `TOP_K` retrieved chunks (default 10; one shown here).

```json
{
  "query": "हिरलूम टमाटर क्या है",
  "transcript": null,
  "answer": "एक पुरानी किस्म जो खुले परागण से उगाई जाती है",
  "sources": [
    {
      "text": "हिरलूम टमाटर एक पुरानी किस्म है ...",
      "score": 0.82,
      "lang": "hi",
      "lang_name": "Hindi",
      "strategy": "passage_native"
    }
  ],
  "confidence": 0.41,
  "grounded": true,
  "guardrail_triggered": null,
  "timings_ms": {
    "embed_ms": 8.1,
    "search_ms": 99.5,
    "rerank_ms": 17.6,
    "qa_ms": 18.0,
    "total_ms": 143.2
  },
  "lang_detected": "hi"
}
```

`nli_ms` appears only when the entailment check actually ran — it is skipped
unless QA confidence falls in the ambiguous 0.15–0.35 band and the request is
still inside its latency budget.

### Declines

When a guardrail fires, `grounded` is `false`, `confidence` is `0.0`, `sources`
is empty, and `guardrail_triggered` names the reason:

| Code                       | Meaning                                       |
| -------------------------- | --------------------------------------------- |
| `unsafe_input`             | Harmful request, refused before any retrieval |
| `out_of_scope`             | Real-time or current-events question          |
| `off_topic`                | No corpus match above the score threshold     |
| `low_retrieval_confidence` | Retrieval scores too weak                     |
| `low_qa_confidence`        | No confident answer span found                |
| `low_answer_relevance`     | Extracted answer unrelated to the question    |
| `script_mismatch`          | Answer script differs from the query script   |
| `implausible_answer`       | Failed the domain plausibility check          |
| `not_entailed`             | NLI entailment verification failed            |
| `stt_failed`               | Audio could not be transcribed                |

Note: errors return HTTP 200 with an `error` key rather than a 4xx/5xx status.

## Known limitations

- Extractive QA accuracy is meaningfully lower on Hindi/Marathi than English (76.7–93.3% vs. 93.3–100%), reflecting the underlying QA model's training data skew toward English SQuAD2 with only cross-lingual transfer to Indic languages
- Very-low-confidence extractions (QA score <0.15) bypass the NLI entailment check by design, to preserve latency headroom — a small number of confidently-wrong-but-low-score extractions may not be caught
- The system answers only from its indexed corpus (MSMARCO-XI + MS MARCO v2.1 derived passages); general knowledge and current-events questions are explicitly declined by design (Guardrail 1b), not hallucinated

## Engineering process archive

`archive/` contains the full experimental history — patch scripts, diagnostic tools, and legacy benchmarks from the development process, including earlier index-scale experiments (58K → 2.3M → 2.76M → 1.5M multi-strategy) and rejected approaches (generative LLM backends via Qwen/Gemini, evaluated and found 4x slower than extractive QA for this latency budget).
