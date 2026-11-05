# Accuracy evaluation: v5-qa-large-sentences

- Deployment: `https://rawrmeinkayanosaurushun--voice-rag-v5-voicerag-fastapi-app.us-east.modal.run`
- Code commit: `620c4db`, run 2026-11-05T07:51:16+00:00
- Questions: 300 in Set A, 300 in Set B, 27 in Set C

## Set A: the gold passage is in the index

| Language | Recall@1 | Recall@5 | MRR | Exact match | F1 | Contains | Refused |
|---|---|---|---|---|---|---|---|
| hi | 12.0% | 26.0% | 0.180 | 0.0% | 18.2% | 4.0% | 23.0% |
| mr | 5.0% | 19.0% | 0.101 | 0.0% | 7.5% | 5.0% | 23.0% |
| en | 30.0% | 53.0% | 0.388 | 3.0% | 26.0% | 23.0% | 6.0% |
| all | 15.7% | 32.7% | 0.223 | 1.0% | 17.2% | 10.7% | 17.3% |

## Set B: held-out questions (never indexed)

| Language | Exact match | F1 | Contains | Answered | F1 when answered | Contains when answered |
|---|---|---|---|---|---|---|
| hi | 0.0% | 13.4% | 3.0% | 75.0% | 17.9% | 4.0% |
| mr | 0.0% | 6.7% | 1.0% | 72.0% | 9.3% | 1.4% |
| en | 2.0% | 17.4% | 5.0% | 79.0% | 22.1% | 6.3% |
| all | 0.7% | 12.5% | 3.0% | 75.3% | 16.6% | 4.0% |

## Set C: guardrail cases (should be declined)

| Expected | Correct | Total |
|---|---|---|
| decline | 6 | 10 |
| out_of_scope | 11 | 11 |
| unsafe_input | 6 | 6 |

## Chunking strategies (Set A)

| Strategy | Top-1 hits on the gold passage | Share of all returned sources |
|---|---|---|
| passage_native | 23 (49%) | 49% |
| fixed_overlap | 14 (30%) | 27% |
| semantic_window | 10 (21%) | 24% |

Server latency over all questions: P50 69.4 ms, P90 92.2 ms.

## How to read this

- **Recall@k / MRR:** is the gold passage (the exact passage the dataset marks as the answer source) among the top k sources, and how high (MRR = average of 1/rank). Declined questions return no sources and count as misses.
- **Exact match / F1 / Contains:** the answer compared with the dataset's answer after normalization (case, punctuation, digits). F1 is word overlap. *Contains* means one answer appears word-for-word inside the other. Dataset answers are often full sentences while this system extracts short spans, so exact match is naturally low.
- Declined questions score 0 on answer metrics. *When answered* columns exclude them.
- Set A is optimistic: the question's own passage is in the index. Set B shows behavior on unseen questions.
- The strategy table counts which chunking strategy supplied the gold passage when it ranked first; it is a contribution count, not a controlled ablation.
