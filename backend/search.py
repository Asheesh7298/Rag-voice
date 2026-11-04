"""
Exact vector search over the FP16 passage vectors held in GPU memory, using CuPy.

Same math as v1 (PyTorch `matmul` + `topk`): one FP16 matrix-vector product scores every passage
against the question, then the k highest scores are returned, best first.
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

DIM = 768
BLOCK = 1024                        # top-k is found block by block, see search()
READ_WORKERS = 16                   # parallel readers for the vector file, see load()
CHUNK_BYTES = 64 * 1024 * 1024      # bytes each reader fetches at a time


class VectorIndex:
    def __init__(self, vectors: cp.ndarray):
        self.vectors = vectors              # [N, 768] float16, in GPU memory
        self.n = int(vectors.shape[0])

    @classmethod
    def load(cls, path: str, dim: int = DIM, workers: int = READ_WORKERS) -> "VectorIndex":
        """Read the raw FP16 file (~20 GB) into GPU memory. The vector count comes from the file size.

        On Modal's network storage, reading it through one memory map (page by page, a single reader) took
        between 10 and 234 seconds. Here many readers fetch large chunks in parallel: each reads 64 MB into
        host memory and copies it into its own slice of a GPU array allocated up front.
        """
        row_bytes = dim * 2
        n = os.path.getsize(path) // row_bytes
        rows_per_chunk = max(1, CHUNK_BYTES // row_bytes)
        vectors = cp.empty((n, dim), dtype=cp.float16)
        fd = os.open(path, os.O_RDONLY)

        def copy_rows(start: int) -> None:
            end = min(start + rows_per_chunk, n)
            host = np.empty((end - start, dim), dtype=np.float16)
            buf, offset, done = memoryview(host.reshape(-1).view(np.uint8)), start * row_bytes, 0
            while done < len(buf):
                got = os.preadv(fd, [buf[done:]], offset + done)
                if got == 0:
                    raise IOError(f"unexpected end of {path} at byte {offset + done}")
                done += got
            vectors[start:end].set(host)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(copy_rows, range(0, n, rows_per_chunk)))
            _spot_check(vectors, np.memmap(path, dtype=np.float16, mode="r", shape=(n, dim)))
        finally:
            os.close(fd)
        return cls(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[list[float], list[int]]:
        """Return (scores, row ids) of the k most similar vectors, highest score first.

        `query` must be L2-normalized (all stored vectors are), so the dot product is cosine similarity.
        """
        k = min(k, self.n)
        scores = self.vectors @ cp.asarray(query.astype(np.float16))      # [N] float16

        # Exact top-k without sorting all N scores: take the best score of every block of 1,024, keep
        # the blocks with the highest block-best, and pick the top k inside them. Every one of the k best
        # scores sits in a block whose best is at least as high, so those blocks contain them all
        # (2k blocks leave a margin for FP16 ties). CuPy's argpartition would sort the whole array.
        n_full = (self.n // BLOCK) * BLOCK
        block_best = scores[:n_full].reshape(-1, BLOCK).max(axis=1)
        if n_full < self.n:
            block_best = cp.concatenate([block_best, scores[n_full:].max(keepdims=True)])
        top_blocks = cp.argsort(-block_best.astype(cp.float32))[: min(2 * k, int(block_best.shape[0]))]
        candidates = (top_blocks[:, None] * BLOCK + cp.arange(BLOCK)[None, :]).ravel()
        candidates = candidates[candidates < self.n]

        ids = cp.asnumpy(candidates)
        vals = cp.asnumpy(scores[candidates]).astype(np.float32)
        order = np.lexsort((ids, -vals))[:k]          # highest score first; ties go to the lower row id
        return vals[order].tolist(), ids[order].tolist()


def _spot_check(vectors: cp.ndarray, on_disk: np.ndarray, samples: int = 64) -> None:
    """Compare random rows (plus the first and last) of the GPU copy with the file, to catch loading bugs."""
    n = on_disk.shape[0]
    rows = [0, n - 1] + random.sample(range(n), min(samples, n))
    for row in rows:
        if not np.array_equal(cp.asnumpy(vectors[row]), on_disk[row]):
            raise RuntimeError(f"vector row {row} in GPU memory differs from the file")
