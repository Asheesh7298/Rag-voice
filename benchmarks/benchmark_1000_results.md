# Official 1,000-Query Latency Benchmark Results

- **Queries**: 1,000
- **Infrastructure**: Modal T4 GPU (min_containers=1, warm container)
- **Status**: All requirements met (P99 < 100 ms)

| Stage | P50 | P90 | P95 | P99 | P99.9 | Mean | Target |
|---|---|---|---|---|---|---|---|
| `embed_ms` | 12.8 ms | 13.5 ms | 14.3 ms | 16.0 ms | 21.0 ms | 13.0 ms | - |
| `search_ms` | 14.6 ms | 15.0 ms | 16.2 ms | 17.0 ms | 19.2 ms | 14.8 ms | - |
| `rerank_ms` | 3.5 ms | 4.2 ms | 4.4 ms | 4.8 ms | 6.7 ms | 3.6 ms | - |
| `qa_ms` | 28.5 ms | 29.4 ms | 29.7 ms | 34.1 ms | 55.2 ms | 28.0 ms | - |
| `total_ms` | 59.6 ms | 61.6 ms | 62.7 ms | 69.9 ms | 86.6 ms | 59.3 ms | ✅ < 100ms |
