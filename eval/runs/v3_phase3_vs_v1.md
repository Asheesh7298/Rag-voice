# Parity: v1-run1 vs v3-phase3

- A: `eval/baseline_v1.json` (commit 729afa0, 2026-11-04T19:41:14+00:00)
- B: `eval/runs/v3_phase3.json` (commit 9ef5a93, 2026-11-04T22:39:36+00:00)
- Questions compared: 224

| Check | Identical | % |
|---|---|---|
| Answer text | 224/224 | 100.0% |
| Guardrail decision | 224/224 | 100.0% |
| Top source | 224/224 | 100.0% |
| Top-8 sources (as a set) | 224/224 | 100.0% |
| HTTP status | 224/224 | 100.0% |

## Latency (ms)

| Stage | A P50 | B P50 | A P90 | B P90 | A max | B max |
|---|---|---|---|---|---|---|
| embed_ms | 14.1 | 4.1 | 15.5 | 4.5 | 17.1 | 6.0 |
| search_ms | 19.2 | 23.3 | 22.1 | 25.4 | 25.4 | 29.3 |
| rerank_ms | 2.7 | 3.1 | 3.5 | 4.0 | 8.1 | 10.2 |
| qa_ms | 19.1 | 17.8 | 23.6 | 25.2 | 60.1 | 35.4 |
| total_ms | 59.1 | 50.2 | 72.1 | 60.1 | 97.4 | 75.3 |
| client_round_trip_ms | 1327.3 | 1377.0 | 1436.8 | 1443.6 | 9868.7 | 1826.5 |

Server stages come from the response's `timings_ms` (questions stopped early by a guardrail have fewer stages). `client_round_trip_ms` also includes the network to Modal.

## Differences (0)

