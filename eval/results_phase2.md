# Phase 2 results: accuracy evaluation of the current system

Measured on `voice-rag-v2` (Phase 1 cleanup, answers identical to the live v1), 2026-11-04 (UTC).
Full report: [results_eval_v2-phase1.md](results_eval_v2-phase1.md). Numbers: [results_eval_v2-phase1.json](results_eval_v2-phase1.json).

## Evaluation sets ([sets/](sets/), built by [build_eval_sets.py](build_eval_sets.py), seed 42)

| Set | What it is | Questions |
|---|---|---|
| A | Training questions whose gold (selected) passage is in the index | 100 Hindi, 100 Marathi, 100 English |
| B | Validation questions with a known answer, never indexed | 100 Hindi, 100 Marathi, 100 English |
| C | Unsafe, current-events and off-topic test questions | 27 |

## Headline numbers

| | Hindi | Marathi | English | All |
|---|---|---|---|---|
| Set A Recall@5 (gold passage in the top 5 sources) | 22% | 15% | 47% | 28% |
| Set A Recall@1 | 13% | 5% | 31% | 16% |
| Set A answer F1 | 13% | 10% | 25% | 16% |
| Set B answer F1 | 9% | 6% | 18% | 11% |
| Set B questions answered (not refused) | 99% | 96% | 99% | 98% |

Set C: all 11 current-events and all 6 English unsafe questions were declined; only 2 of 10 other off-topic or
non-English unsafe questions were declined.

## What the misses show

Looking at Set A questions whose gold passage was not retrieved:

1. **Duplicate chunks crowd the results.** A short passage is stored three times (one per chunking strategy) with
   identical text, so one passage often fills 3 of the 8 source slots.
2. **The language filter runs after a small candidate pool.** Search takes the top ~35 matches across all
   languages, then keeps only the question's language. Hindi and Marathi translations of the same passage sit
   very close together, so a Hindi question can be left with only one to three Hindi candidates.
3. **The system almost never refuses.** It answered 98% of held-out questions, mostly with low-overlap answers,
   and the off-topic gate did not fire (the scores it compares are reranked scores, which are inflated).
4. **Answers often repeat the question.** Found while porting the code in Phase 3: v1's step that keeps answer
   spans out of the question tokens never ran on the GPU, so many answers start with the question text.

These are fixed or tuned in Phase 6, using these same sets to measure the effect.

## How to read the metrics

- Recall@k and MRR use the exact gold passage, so they are strict; matching any passage gathered for the same
  training question gives Recall@5 of 34% overall (27% Hindi, 19% Marathi, 56% English).
- Dataset answers are often full sentences while the system extracts short spans, so exact match is naturally
  low; F1 (word overlap) and "contains" are more informative.
- No human labeled the answers: the scores are automatic comparisons with the dataset's answers.
