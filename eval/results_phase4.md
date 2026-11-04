# Phase 4 results: shorter, more reliable loading screen

Deployed to `voice-rag-v3` on 2026-11-04 (UTC). The server still scales to zero when idle.

## Changes

1. **Separate startup limit (`startup_timeout=600`).** In Phase 0 one start took ~331 s: the vector load alone
   took 234 s, the start passed the 300 s limit (`timeout=300` also applied to startup), and Modal killed the
   container and began again from zero. Startup now has its own 600 s limit, so a slow start finishes; the
   300 s limit for answering a request is unchanged.
2. **Parallel vector loading** ([backend/search.py](../backend/search.py)). The 20 GB vector file is read by 16
   workers in 64 MB chunks, each copied into its slice of a GPU array allocated up front, instead of one reader
   going page by page through a memory map. After loading, 66 rows (first, last and 64 random) are compared with
   the file; a mismatch stops the server from starting.

## Measurements

| | Vector load | Loading-screen wait |
|---|---|---|
| Earlier starts, memory map (Phases 0–3) | 10.4, 10.8, 10.8, 18.7, 22.7, 26.6, 27.4 and 234 s | 25.9–53.3 s, and ~331 s once |
| Parallel loader, cold start | 14.4 s (spot check passed) | 26.8 s |

One sample is not enough to claim a speed-up: on a normal day both methods seem limited by the storage itself
(about 1.4–1.9 GB/s). The parallel reads are meant to help on slow-storage days, which could not be reproduced
on demand. Further cold-start measurements were cancelled to save Modal credits.

Answers: the loader only changes how the vectors reach the GPU, and the spot check confirms they arrive
unchanged. The parity run made right after this deploy was answered by the previous warm container (stopping it
failed without `--yes`), so it does not count as a check of this code; the Phase 6 runs use this loader.
