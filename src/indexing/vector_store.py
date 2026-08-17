"""
Thin FAISS wrapper. Uses HNSW (fast approximate search, no training step needed)
instead of brute-force IndexFlatIP or an IVF-PQ index that needs a training pass.
HNSW gives sub-10ms search at our scale (tens of thousands of vectors) which is
what keeps the retrieval stage inside the <200ms budget.
"""
from __future__ import annotations
import json
import os
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.metadata: list[dict] = []  # parallel array, row i <-> vector i

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        assert vectors.shape[0] == len(metadatas)
        assert vectors.shape[1] == self.dim
        # vectors must be L2-normalized upstream so inner product == cosine similarity
        self.index.add(vectors.astype(np.float32))
        self.metadata.extend(metadatas)

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[dict, float]]:
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        scores, ids = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, in_dir: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(in_dir, "index.faiss"))
        store.metadata = []
        with open(os.path.join(in_dir, "metadata.jsonl"), encoding="utf-8") as f:
            for line in f:
                store.metadata.append(json.loads(line))
        return store
