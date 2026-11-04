# Parity: v1-run1 vs v2-phase1

- A: `eval/baseline_v1.json` (commit 729afa0, 2026-11-04T19:41:14+00:00)
- B: `eval/runs/v2_phase1.json` (commit 510dfbf, 2026-11-04T21:32:26+00:00)
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
| embed_ms | 14.1 | 13.2 | 15.5 | 14.5 | 17.1 | 15.9 |
| search_ms | 19.2 | 18.7 | 22.1 | 20.1 | 25.4 | 22.8 |
| rerank_ms | 2.7 | 3.1 | 3.5 | 4.0 | 8.1 | 9.6 |
| qa_ms | 19.1 | 21.6 | 23.6 | 26.8 | 60.1 | 37.5 |
| total_ms | 59.1 | 61.6 | 72.1 | 73.9 | 97.4 | 84.2 |
| client_round_trip_ms | 1327.3 | 1366.9 | 1436.8 | 1438.5 | 9868.7 | 1833.8 |

Server stages come from the response's `timings_ms` (questions stopped early by a guardrail have fewer stages). `client_round_trip_ms` also includes the network to Modal.

## Differences (0)

