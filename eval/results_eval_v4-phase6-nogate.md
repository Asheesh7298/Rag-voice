# Accuracy evaluation: v4-phase6-nogate

- Deployment: `https://rawrmeinkayanosaurushun--voice-rag-v4-voicerag-fastapi-app.us-east.modal.run`
- Code commit: `e7b35c7`, run 2026-11-04T23:24:29+00:00
- Questions: 300 in Set A, 300 in Set B, 27 in Set C

## Set A: the gold passage is in the index

| Language | Recall@1 | Recall@5 | MRR | Exact match | F1 | Contains | Refused |
|---|---|---|---|---|---|---|---|
| hi | 12.0% | 27.0% | 0.186 | 2.0% | 13.0% | 13.0% | 3.0% |
| mr | 6.0% | 20.0% | 0.111 | 1.0% | 8.3% | 5.0% | 6.0% |
| en | 31.0% | 55.0% | 0.402 | 7.0% | 26.1% | 29.0% | 3.0% |
| all | 16.3% | 34.0% | 0.233 | 3.3% | 15.8% | 15.7% | 4.0% |

## Set B: held-out questions (never indexed)

| Language | Exact match | F1 | Contains | Answered | F1 when answered | Contains when answered |
|---|---|---|---|---|---|---|
| hi | 1.0% | 7.4% | 4.0% | 94.0% | 7.8% | 4.3% |
| mr | 0.0% | 3.2% | 2.0% | 91.0% | 3.5% | 2.2% |
| en | 1.0% | 14.0% | 9.0% | 96.0% | 14.6% | 9.4% |
| all | 0.7% | 8.2% | 5.0% | 93.7% | 8.7% | 5.3% |

## Set C: guardrail cases (should be declined)

| Expected | Correct | Total |
|---|---|---|
| decline | 5 | 10 |
| out_of_scope | 11 | 11 |
| unsafe_input | 6 | 6 |

## Chunking strategies (Set A)

| Strategy | Top-1 hits on the gold passage | Share of all returned sources |
|---|---|---|
| passage_native | 25 (51%) | 47% |
| fixed_overlap | 14 (29%) | 27% |
| semantic_window | 10 (20%) | 26% |

Server latency over all questions: P50 46.6 ms, P90 56.3 ms.

## How to read this

- **Recall@k / MRR:** is the gold passage (the exact passage the dataset marks as the answer source) among the top k sources, and how high (MRR = average of 1/rank). Declined questions return no sources and count as misses.
- **Exact match / F1 / Contains:** the answer compared with the dataset's answer after normalization (case, punctuation, digits). F1 is word overlap. *Contains* means one answer appears word-for-word inside the other. Dataset answers are often full sentences while this system extracts short spans, so exact match is naturally low.
- Declined questions score 0 on answer metrics. *When answered* columns exclude them.
- Set A is optimistic: the question's own passage is in the index. Set B shows behavior on unseen questions.
- The strategy table counts which chunking strategy supplied the gold passage when it ranked first; it is a contribution count, not a controlled ablation.
