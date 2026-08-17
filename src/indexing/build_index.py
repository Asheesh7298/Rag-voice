"""
Day 2: read data/processed/passages.jsonl, run all chunking strategies over every
passage, embed the resulting chunks, and build one merged FAISS HNSW index tagged
with `chunk_strategy` metadata (so retrieval.py can filter/compare by strategy, and
you can run an ablation: "which strategy contributes the most hits?").

Run: python -m src.indexing.build_index
"""
import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.chunking.strategies import chunk_all_strategies, Chunk

PASSAGES_PATH = "data/processed/passages.jsonl"


def load_passages(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_embed_fn(model: SentenceTransformer):
    def embed_fn(texts: list[str]) -> np.ndarray:
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)
    return embed_fn


def main():
    print(f"Loading embedding model: {settings.embed_model}")
    model = SentenceTransformer(settings.embed_model)
    embed_fn = make_embed_fn(model)

    print(f"Loading passages from {PASSAGES_PATH}")
    passages = load_passages(PASSAGES_PATH)
    print(f"  {len(passages)} passages loaded")

    all_chunks: list[Chunk] = []
    for row in tqdm(passages, desc="Chunking"):
        all_chunks.extend(chunk_all_strategies(row, embed_fn))

    print(f"Total chunks across all strategies: {len(all_chunks)}")
    by_strategy = {}
    for c in all_chunks:
        by_strategy[c.strategy] = by_strategy.get(c.strategy, 0) + 1
    print("Breakdown:", by_strategy)

    # Embed in batches for speed
    texts = [c.text for c in all_chunks]
    print("Embedding all chunks (this is the slow, one-time offline step)...")
    batch_size = 256
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        vecs.append(embed_fn(batch))
    vectors = np.vstack(vecs)

    from src.indexing.vector_store import VectorStore
    store = VectorStore(dim=vectors.shape[1])
    metadatas = [{
        "chunk_id": c.id,
        "text": c.text,
        "strategy": c.strategy,
        "lang": c.lang,
        "query_id": c.query_id,
        "source_passage_id": c.source_passage_id,
        "is_selected": c.is_selected,
        **c.extra,
    } for c in all_chunks]

    store.add(vectors, metadatas)
    store.save(settings.index_dir)
    print(f"Index saved to {settings.index_dir}")
    print(f"  vectors: {vectors.shape[0]}, dim: {vectors.shape[1]}")


if __name__ == "__main__":
    main()
