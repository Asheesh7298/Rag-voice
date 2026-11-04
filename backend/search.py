"""
Exact vector search over the FP16 passage vectors held in GPU memory, using CuPy.

Same math as v1 (PyTorch `matmul` + `topk`): one FP16 matrix-vector product scores every passage
against the question, then the k highest scores are returned, best first.
"""
from __future__ import annotations

import os

import cupy as cp
import numpy as np

DIM = 768
BLOCK = 1024   # top-k is found block by block, see search()


class VectorIndex:
    def __init__(self, vectors: cp.ndarray):
        self.vectors = vectors              # [N, 768] float16, in GPU memory
        self.n = int(vectors.shape[0])

    @classmethod
    def load(cls, path: str, dim: int = DIM) -> "VectorIndex":
        """Read the raw FP16 file into GPU memory. The vector count comes from the file size."""
        n = os.path.getsize(path) // (dim * 2)
        host = np.memmap(path, dtype=np.float16, mode="r", shape=(n, dim))
        return cls(cp.asarray(host))

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
