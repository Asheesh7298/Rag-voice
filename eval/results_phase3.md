# Phase 3 results: PyTorch → ONNX Runtime + CuPy

Deployed as a **separate** Modal app, `voice-rag-v3`, on 2026-11-04 (UTC). The live app `voice-rag` was not touched.
The server image no longer contains PyTorch, Transformers, sentence-transformers or FAISS.

## What changed

- **Models run with ONNX Runtime** ([backend/encoders.py](../backend/encoders.py)), GPU via the CUDA 13 libraries
  from pip, TF32 off (PyTorch's default full-FP32 math).
  - e5: the official ONNX export published by its authors (`onnx/model.onnx`, revision `d128750`).
  - QA (`deepset/xlm-roberta-base-squad2`, revision `a5fab99`): converted to ONNX once in a temporary Modal job
    (kept outside this repo) and stored on the Volume at `onnx/xlm-roberta-base-squad2/`, where
    `export_info.json` records the versions used and the check against the original model
    (largest logit difference 2.6e-5 on Hindi, Marathi and English examples; same best answer positions).
- **Vector search runs with CuPy** ([backend/search.py](../backend/search.py)): the same FP16 matrix-vector product,
  then an exact top-k that finds the best blocks first instead of sorting all 13M scores.
- **Tokenization** uses `tokenizers` 0.19.1 directly, the version v1 used through Transformers, with the same
  truncation, padding and decoding (including Transformers' cleanup of spaces before punctuation).
- Answer-span math moved from PyTorch tensors to NumPy.

## A v1 bug found while porting

v1's step that keeps answer spans out of the question tokens never ran on the GPU: moving the tokenizer output
to CUDA turned it into a plain dict without `.sequence_ids()`, and the error was swallowed by `except: pass`.
That is why many v1 answers start by repeating the question. v3 reproduces this exactly behind
`MASK_QUESTION_TOKENS = False` in `modal_app.py`; turning it on is a deliberate change for Phase 6.

## Checks

| Check | Result | Pass bar |
|---|---|---|
| Stored-vector check: ONNX e5 vs the PyTorch vectors in the index, 1,000 random passages ([check_embeddings.py](check_embeddings.py)) | average cosine 0.9999975, lowest 0.99998 | ≥ 0.999 average, ≥ 0.995 lowest |
| Parity with the v1 baseline, 224 questions ([runs/v3_phase3_vs_v1.md](runs/v3_phase3_vs_v1.md)) | 224/224 identical answers, guardrail decisions, top source, top-8 sources | every difference approved |

The first parity runs showed differences: 21 from the masking bug above, then a run that was split between an
old warm container and the newly deployed one. Lesson: after redeploying, stop old containers before comparing.

## Latency (server, milliseconds, 224 questions)

| Stage | v1 P50 | v3 P50 | v1 P90 | v3 P90 |
|---|---|---|---|---|
| Embedding | 14.1 | 4.1 | 15.5 | 4.5 |
| Search | 19.2 | 23.3 | 22.1 | 25.4 |
| Rerank | 2.7 | 3.1 | 3.5 | 4.0 |
| QA | 19.1 | 17.8 | 23.6 | 25.2 |
| **Total** | **59.1** | **50.2** | **72.1** | **60.1** |

Total is 15% faster at P50 and 17% at P90 (max 75.3 ms vs 97.4 ms). Embedding is 3.5x faster; search is
about 4 ms slower than PyTorch's `topk`, which is a possible later optimization.
