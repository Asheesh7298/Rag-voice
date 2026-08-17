"""
Quick test to verify whether adding e5 prefixes fixes retrieval.
Run locally: python scripts/test_retrieval.py
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

print("Loading model and index...")
model = SentenceTransformer("intfloat/multilingual-e5-base")
model.max_seq_length = 64

index = faiss.read_index("data/index/index.faiss")
metadata = []
with open("data/index/metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

query = "हिरलूम टमाटर का क्या अर्थ है"

print("\n--- WITHOUT prefix ---")
vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32).reshape(1, -1)
scores, ids = index.search(vec, 5)
for score, idx in zip(scores[0], ids[0]):
    m = metadata[idx]
    print(f"  score={score:.4f} lang={m['lang']} text={m['text'][:80]}")

print("\n--- WITH 'query: ' prefix ---")
vec2 = model.encode([f"query: {query}"], normalize_embeddings=True)[0].astype(np.float32).reshape(1, -1)
scores2, ids2 = index.search(vec2, 5)
for score, idx in zip(scores2[0], ids2[0]):
    m = metadata[idx]
    print(f"  score={score:.4f} lang={m['lang']} text={m['text'][:80]}")