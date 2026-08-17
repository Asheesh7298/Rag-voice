# Voice-RAG — Indic MSMARCO

Voice-enabled RAG: mic → STT (Sarvam) → multi-strategy chunked retrieval (FAISS) → grounded answer generation, with guardrails and full latency instrumentation.

## Architecture

```
Mic audio
  -> Sarvam STT                      [network call, timed separately]
  -> Guardrail: off-topic / unsafe gate
  -> Query embed -> FAISS ANN search -> rerank -> context assemble   [THIS is the <200ms budget]
  -> LLM generation (structured JSON output)                        [network call, timed separately]
  -> Guardrail: groundedness / hallucination check
  -> Response {answer, sources, confidence, grounded}
```

**Latency scoping (read this before benchmarking):** the <200ms target applies to
chunking + vector retrieval + context assembly — the part of the pipeline that's actually
under our control and doesn't depend on a third-party network round trip. STT and LLM
generation calls are instrumented and reported *separately* (P50/P70/P100 each) rather than
folded into one misleading end-to-end number. See `benchmarks/latency_bench.py` and
`benchmarks/results.md` after running it.

## Repo layout

```
data/download_data.py        # pulls IndicMSMARCO (all 13 langs, subsampled) from HF
src/chunking/strategies.py   # 3 chunking strategies + metadata tagging
src/indexing/build_index.py  # embeds chunks, builds FAISS HNSW index per strategy
src/indexing/vector_store.py # FAISS wrapper: save/load/search
src/retrieval/retriever.py   # query embed -> search -> rerank -> assemble context
src/stt/sarvam_client.py     # Sarvam STT wrapper with retries
src/generation/llm_client.py # LLM call with structured JSON schema + retries
src/guardrails/checks.py     # off-topic gate, confidence gate, groundedness check
src/harness/pipeline.py      # the state machine that wires everything together
src/harness/schemas.py       # Pydantic schemas for structured I/O at every stage
src/api/main.py              # FastAPI app exposing /query (text) and /voice-query (audio)
frontend/index.html          # minimal mic-recorder + chat UI, no build step
benchmarks/latency_bench.py  # runs N queries, logs per-stage timings, computes P50/P70/P100
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY and your LLM API key
python data/download_data.py
python -m src.indexing.build_index
uvicorn src.api.main:app --reload --port 8000
# open frontend/index.html in a browser, or serve it statically
```

## Benchmarking

```bash
python -m benchmarks.latency_bench --n 60
```

Writes `benchmarks/results.md` with a P50/P70/P100 table per stage. Run this with a
real, varied query set (not the same 3 queries repeated) — mix languages and query lengths.

## Guardrail behavior (what "knowing when not to answer" looks like)

- **Off-topic gate**: if the query's embedding similarity to the corpus centroid /
  nearest cluster falls below `OFF_TOPIC_THRESHOLD`, the pipeline returns a decline
  response before ever calling the LLM.
- **Unsafe input filter**: lightweight keyword + heuristic pass before retrieval.
- **Retrieval-confidence gate**: if the top-k retrieved chunks all score below
  `MIN_RETRIEVAL_SCORE`, we return "not enough grounded information" instead of
  letting the LLM improvise.
- **Groundedness check**: after generation, we check whether the answer's claims
  are supported by the retrieved context (lexical/semantic overlap check); if not,
  the response is flagged `grounded: false` and suppressed/replaced with a
  fallback message rather than shown as a confident answer.

All four are logged with a trigger reason — surface these logs in the demo video.
