# QA upgrade: large reader model and full-sentence answers

Measured 2026-11-05 on `voice-rag-v5`, then adopted for the live app (`voice-rag-v4`). Compared with the Phase 6
system on the same 627 questions ([results_eval_v5-qa-large-sentences.md](results_eval_v5-qa-large-sentences.md)).

## Why answers were often imperfect (Set A, Phase 6 system)

| Gold passage | Questions | Answer correct* |
|---|---|---|
| Ranked #1 | 49 | 33% |
| Ranked #2–8 | 71 | 28% |
| Not retrieved | 168 (56%) | 7% |
| Declined | 12 | – |

\*the answer appears word for word in the dataset answer. Retrieval is the larger limit; the reader also missed
when it had the right passage, and its short spans read as fragments (dataset answers have a median of 9 words).

## Changes

1. **Reader:** `deepset/xlm-roberta-large-squad2` (revision `dafe599`) replaces the base model. Converted to ONNX once
   (weights in `model.onnx_data`); largest logit difference vs PyTorch 3.7e-5, same best answer positions.
2. **Full-sentence answers:** the answer is the passage sentence containing the extracted span, found from the
   tokenizer's character offsets; the span is returned as `answer_span` and shown in bold in the web app.
   Guardrails still check the span.

## Results

| | Before | After |
|---|---|---|
| Answer F1 on questions both versions answered (Set A / Set B) | 18.3% / 9.5% | **21.0% / 16.6%** (sentences); spans alone 20.1% / 11.0% |
| Refused (Set A / Set B) | 4% / 6% | 17% / 25% |
| Server latency P50 / P90 | 47 / 56 ms | 69 / 92 ms (QA step 19 → 43 ms) |
| Set C declined | 22 / 27 | 23 / 27 |

**The extra refusals are mostly good ones.** All come from `low_qa_confidence`: the large model, trained on
SQuAD 2.0, gives more weight to "no answer in this passage". Of the 40 Set A questions it newly refuses, the old
system answered only 2 correctly (and had the gold passage for 5); of the 57 Set B questions, only 3. So about 92
confidently wrong answers became "Couldn't extract a confident answer", at the cost of about 5 correct ones.
Refusals are highest in Hindi and Marathi (23% of Set A) and lowest in English (6%).

Retrieval is unchanged. Recall@5 measures 32.7% here only because refused questions return no sources.
