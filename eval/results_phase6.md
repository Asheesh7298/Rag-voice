# Phase 6 results: behavior fixes (v4)

Deployed as a **separate** Modal app, `voice-rag-v4`, on 2026-11-04 (UTC); `voice-rag` (live) and `voice-rag-v3`
(same answers as v1) are unchanged. Evaluated with the Phase 2 sets
([results_eval_v4-phase6-nogate.md](results_eval_v4-phase6-nogate.md)); 627 questions, 8 in flight, 171 s of GPU time.

## Changes

| Change | Changes answers? |
|---|---|
| Search only the question's language block (the index stores hi, mr, en rows as contiguous blocks; found by binary search at startup and spot-checked) | yes |
| Remove duplicate passages (identical text stored once per chunking strategy) from the results | yes |
| Keep answer spans out of the question tokens (`MASK_QUESTION_TOKENS = True`, the v1 bug) | yes |
| Hindi/Marathi unsafe-input check (weapon/explosive word + "make" verb, or self-harm phrasing; matched by word start) | yes, for those questions |
| Raw-cosine off-topic gate (`OFF_TOPIC_COSINE`), reported as `retrieval_score` | no: left off, see below |
| `strategy` field reads the stored `chunk_strategy` | API field only |
| `/query` returns HTTP 400 / 500 with no stack trace | errors only |
| Voice uploads forwarded to Sarvam with their real file name and type (not always `audio/wav`) | voice only (not tested with real speech) |
| One container at most (`max_containers=1`), up to 8 requests queued on it (`@modal.concurrent`) | no |
| Frontend retries network errors and HTTP 5xx twice (3 s, 6 s) | no |

The Hindi/Marathi unsafe check flags exactly the two unsafe test questions and none of the other 822 questions in
the evaluation and parity sets.

## Before (v2 = v1 answers) and after (v4)

| | Before | After |
|---|---|---|
| Set A Recall@5 (gold passage in top 5) | 28.0% (hi 22, mr 15, en 47) | **34.0%** (hi 27, mr 20, en 55) |
| Set A MRR | 0.205 | **0.233** |
| Set A Recall@1 | 16.3% | 16.3% |
| Set A answer F1 / contains | 15.9% / 13.7% | 15.8% / **15.7%** |
| Set B answer F1 / contains | 11.2% / 3.3% | 8.2% / **5.0%** |
| Set B answered (not refused) | 98.0% | 93.7% |
| Set C: non-English unsafe and off-topic declined | 2/10 | **5/10** |
| Set C: current events, English unsafe declined | 17/17 | 17/17 |
| Server latency P50 / P90 | 60.9 / 74.2 ms | **46.6 / 56.3 ms** |

**Answers:** 306 of 600 changed. Judged by "contains": 20 better, 9 worse, 22 newly declined, 6 newly answered.
198 of 600 v1 answers began by repeating the whole question; v4 answers are shorter (about 4.8 words vs 7.9).
Word precision held (Set A 0.24 → 0.26, Set B 0.19 → 0.18) while recall against the long dataset answers fell
(Set B 0.12 → 0.08), which is most of the F1 drop. Some v4 answers are terse single words.

## Off-topic gate: does not work with these scores

| Raw cosine of the best match | Value |
|---|---|
| Answerable questions (Set A), median | 0.884 (5th percentile 0.850) |
| The 8 off-topic questions | 0.860 – 0.918 |

The off-topic questions score as high as answerable ones: e5's similarities are compressed into a narrow band.
Declining even 1 of the 8 would also refuse 11% of answerable questions, so `OFF_TOPIC_COSINE` stays 0 (off).
Detecting off-topic questions needs another signal (e.g. a cross-encoder or a small classifier).
