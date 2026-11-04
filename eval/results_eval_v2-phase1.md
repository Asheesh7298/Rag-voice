# Accuracy evaluation: v2-phase1

- Deployment: `https://rawrmeinkayanosaurushun--voice-rag-v2-voicerag-fastapi-app.us-east.modal.run`
- Code commit: `88ab5e7`, run 2026-11-04T22:09:54+00:00
- Questions: 300 in Set A, 300 in Set B, 27 in Set C

## Set A: the gold passage is in the index

| Language | Recall@1 | Recall@5 | MRR | Exact match | F1 | Contains | Refused |
|---|---|---|---|---|---|---|---|
| hi | 13.0% | 22.0% | 0.161 | 2.0% | 13.1% | 6.0% | 2.0% |
| mr | 5.0% | 15.0% | 0.086 | 1.0% | 9.8% | 6.0% | 5.0% |
| en | 31.0% | 47.0% | 0.368 | 4.0% | 24.8% | 29.0% | 2.0% |
| all | 16.3% | 28.0% | 0.205 | 2.3% | 15.9% | 13.7% | 3.0% |

## Set B: held-out questions (never indexed)

| Language | Exact match | F1 | Contains | Answered | F1 when answered | Contains when answered |
|---|---|---|---|---|---|---|
| hi | 0.0% | 9.2% | 2.0% | 99.0% | 9.3% | 2.0% |
| mr | 0.0% | 6.3% | 1.0% | 96.0% | 6.5% | 1.0% |
| en | 1.0% | 18.2% | 7.0% | 99.0% | 18.4% | 7.1% |
| all | 0.3% | 11.2% | 3.3% | 98.0% | 11.5% | 3.4% |

## Set C: guardrail cases (should be declined)

| Expected | Correct | Total |
|---|---|---|
| decline | 2 | 10 |
| out_of_scope | 11 | 11 |
| unsafe_input | 6 | 6 |

## Chunking strategies (Set A)

| Strategy | Top-1 hits on the gold passage | Share of all returned sources |
|---|---|---|
| passage_native | 24 (49%) | 30% |
| fixed_overlap | 14 (29%) | 35% |
| semantic_window | 11 (22%) | 35% |

Server latency over all questions: P50 60.9 ms, P90 74.2 ms.

## How to read this

- **Recall@k / MRR:** is the gold passage (the exact passage the dataset marks as the answer source) among the top k sources, and how high (MRR = average of 1/rank). Declined questions return no sources and count as misses.
- **Exact match / F1 / Contains:** the answer compared with the dataset's answer after normalization (case, punctuation, digits). F1 is word overlap. *Contains* means one answer appears word-for-word inside the other. Dataset answers are often full sentences while this system extracts short spans, so exact match is naturally low.
- Declined questions score 0 on answer metrics. *When answered* columns exclude them.
- Set A is optimistic: the question's own passage is in the index. Set B shows behavior on unseen questions.
- The strategy table counts which chunking strategy supplied the gold passage when it ranked first; it is a contribution count, not a controlled ablation.
