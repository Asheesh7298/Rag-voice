# Parity: v1-run1 vs v1-run2

- A: `eval/runs/v1_run1.json` (commit 729afa0, 2026-11-04T19:41:14+00:00)
- B: `eval/runs/v1_run2.json` (commit 729afa0, 2026-11-04T19:46:22+00:00)
- Questions compared: 224

| Check | Identical | % |
|---|---|---|
| Answer text | 222/224 | 99.1% |
| Guardrail decision | 222/224 | 99.1% |
| Top source | 222/224 | 99.1% |
| Top-8 sources (as a set) | 222/224 | 99.1% |
| HTTP status | 222/224 | 99.1% |

## Latency (ms)

| Stage | A P50 | B P50 | A P90 | B P90 | A max | B max |
|---|---|---|---|---|---|---|
| embed_ms | 14.1 | 14.6 | 15.5 | 15.3 | 17.1 | 18.4 |
| search_ms | 19.2 | 13.8 | 22.1 | 18.8 | 25.4 | 21.7 |
| rerank_ms | 2.7 | 2.8 | 3.5 | 3.5 | 8.1 | 8.1 |
| qa_ms | 19.1 | 19.1 | 23.6 | 23.2 | 60.1 | 320.0 |
| total_ms | 59.1 | 55.6 | 72.1 | 67.4 | 97.4 | 356.4 |
| client_round_trip_ms | 1327.3 | 1246.0 | 1436.8 | 1433.6 | 9868.7 | 34370.8 |

Server stages come from the response's `timings_ms` (questions stopped early by a guardrail have fewer stages). `client_round_trip_ms` also includes the network to Modal.

## Differences (2)

### b2-en-21 (benchmark, en): differs in raw response differs
- Query: 'what are words with the same pronunciation called'
- A: 'What are words with the same pronunciation called Noun. 1. homophony' (guardrail=None)
- B: 'modal-http: internal error: status InternalFailure: Server has lost track of input\n' (guardrail=None)

### b2-en-22 (benchmark, en): differs in raw response differs
- Query: 'what blood type conflict causes newborn jaundice'
- A: 'Blood type mismatch between the mother and baby' (guardrail=None)
- B: 'modal-http: internal error: status InternalFailure: Server has lost track of input\n' (guardrail=None)

