# Phase 0 results: baseline of the current system (v1, PyTorch)

Measured 2026-11-04 (UTC) against the live Modal app `voice-rag`, deployment `v1` (2026-08-25).
That deployment runs exactly the code tagged `v1-pytorch` (commit `729afa0`).

## What was captured

- **Question set:** [queries_parity.jsonl](queries_parity.jsonl), 224 questions:
  180 benchmark, 5 UI starters, 4 open-ended, 8 unsafe, 11 current events, 8 off-topic, 8 edge cases.
- **Baseline:** [baseline_v1.json](baseline_v1.json), the full response to every question (run 1).
  Every later version is compared against this file with `parity_check.py compare`.
- **Repeatability:** a second run ([runs/v1_noise.md](runs/v1_noise.md)) gave the same answer, guardrail
  decision, top source and top-8 sources for **all 222 questions answered in both runs**.
  The other 2 failed in run 2 because Modal preempted the GPU mid-run (infrastructure, not the code).
  So v1 has no natural variation: any difference in a later version is a real change.

## Server latency (run 1, milliseconds)

| Stage | P50 | P90 | Max |
|---|---|---|---|
| Embedding the question | 14.1 | 15.5 | 17.1 |
| Vector search (13M) | 19.2 | 22.1 | 25.4 |
| Reranking | 2.7 | 3.5 | 8.1 |
| Answer extraction | 19.1 | 23.6 | 60.1 |
| **Total (server)** | **59.1** | **72.1** | **97.4** |

Run 2 total: P50 55.6, P90 67.4 (its max of 356 ms was the first question after the preemption restart).

Round trip from India, as measured by the script: P50 about 1.25–1.33 s. The script opens a new
connection for every question; browsers reuse connections, so the real page is faster, but the
network to Modal is still most of what users wait for.

## Loading-screen wait (cold starts)

Time from the first `/health` call until the server is ready, after 9+ idle minutes.

| # | Wait | Container start + imports | Embedding model | Vector load | Notes |
|---|---|---|---|---|---|
| 1 | 34.5 s | ~14 s | 3 s | 10.8 s | |
| 2 | 43.7 s | ~7 s | 3 s | 26.6 s | |
| 3 | ~331 s | ~16 s | 23 s | **234 s** | Startup passed Modal's 300 s limit; the container was killed and a new one started (vectors 10 s). Reconstructed from Modal logs |

The container that replaced the preempted one during run 2 loaded the vectors in 27.4 s.

**Takeaways**
- Loading the 20 GB vector file usually takes 10–27 s, but storage speed on Modal varies and it took 234 s once.
- `timeout=300` in `modal_app.py` also limits startup. A slow load gets killed and startup begins again,
  which is how one wait reached about 5½ minutes.
- The unused NLI model costs about 1–2 s per start.

## Behavior recorded as-is (to fix in Phase 6, not before)

The new version must first reproduce these, so the comparison stays fair.

- Guardrail outcomes: 202 answered, 11 `out_of_scope`, 6 `unsafe_input`, 5 `low_qa_confidence`, 0 `off_topic`.
- **The off-topic gate never fired.** Examples:
  "tell me a joke" → "Send us a Letter"; "what is my name" → song lyrics;
  "मेरा पसंदीदा रंग कौन सा है" (what is my favourite colour) → "बैंगनी" (purple).
- **The unsafe filter only works in English.** The Hindi and Marathi versions of "how to make a bomb" were answered.
- All 11 current-events questions were correctly declined.
- An empty question returns HTTP 200 with `{"error": "Missing query parameter"}`.
- Open-ended: "tell me about columbus ohio" → "But i hate ohio."

## What this changes in the plan

- **Phase 4:** the slow tail (minutes) matters more than the typical 35–45 s. Priorities:
  faster parallel vector loading, a separate and larger startup limit so a slow load isn't killed,
  then 8-bit vectors (half the data to read) if still needed.
- **Phase 6:** the frontend should retry when the server errors (e.g. during a GPU preemption).
- `parity_check.py` now retries infrastructure errors so they don't count as answer differences.

## How to reproduce

```
python eval/parity_check.py capture --url <deployment URL> --out eval/runs/<name>.json --label <name>
python eval/parity_check.py compare eval/baseline_v1.json eval/runs/<name>.json --report eval/runs/<name>.md
python eval/coldstart.py --url <deployment URL> --runs 1 --idle 540 --out eval/runs/coldstart_<name>.jsonl
```
