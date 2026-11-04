# Phase 1 results: cleanup without behavior change

Deployed as a **separate** Modal app, `voice-rag-v2`, on 2026-11-04 (UTC). The live app `voice-rag` was not touched.

## What changed in `modal_app.py`

- Removed FAISS (import, thread setup, and the fallback search that crashed because no FAISS index was ever loaded).
- Removed the NLI model (`mDeBERTa-v3`): it was downloaded, loaded and warmed on every start but never called.
- Removed dead code: `_extract_answer`, `_generate_answer` (referred to a Qwen model that is never loaded),
  `_check_entailment`, the `/debug-index` and `/debug-qa` endpoints, unused imports.
- Pinned every package to the exact version running in the v1 container (`pip freeze`), and pinned both
  Hugging Face models to fixed revisions, so a rebuild doesn't silently pick up newer libraries.
- Added `chunk_id` and `query_id` to each source in the response (new fields only; needed by the Phase 2 evaluation).

Also removed from the repo: `src/` (an earlier design that nothing imported), `data/download_data.py`,
nine superseded index-building scripts, `requirements.txt` (described the deleted `src/` code), and the
outdated `.vscode/launch.json`. The README now only states what the code does and what was measured.

## Parity with the v1 baseline ([runs/v2_phase1_vs_v1.md](runs/v2_phase1_vs_v1.md))

| Check | Identical |
|---|---|
| Answer text | 224/224 |
| Guardrail decision | 224/224 |
| Top source | 224/224 |
| Top-8 sources | 224/224 |
| HTTP status | 224/224 |

## Latency (server, milliseconds)

| | v1 baseline | v2 | Difference | v1 run-to-run variation |
|---|---|---|---|---|
| P50 | 59.1 | 61.6 | +2.5 | 3.5 |
| P90 | 72.1 | 73.9 | +1.8 | 4.7 |
| Max | 97.4 | 84.2 | −13.2 | |

The differences are within the variation between two runs of the unchanged v1, and the code path a question
takes is identical, so latency is unchanged.

## Loading-screen wait ([runs/coldstart_v2_phase1.jsonl](runs/coldstart_v2_phase1.jsonl))

| # | Wait | Container start + imports | Embedding model | Vector load |
|---|---|---|---|---|
| 1 | 25.9 s | ~8 s | 2 s | 10.8 s |
| 2 | 26.2 s | ~8 s | 2 s | 10.4 s |

v1 measured 34.5 s, 43.7 s and ~331 s ([results_phase0.md](results_phase0.md)). v2 skips one model load and
some imports, and both v2 starts also got fast storage (vectors in ~10.5 s), so two samples can't separate
those effects. The first start of the brand-new v2 image took 53.3 s (one-time GPU setup: warm-up 8.2 s instead of ~0.3 s).
