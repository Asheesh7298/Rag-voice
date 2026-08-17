import json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("intfloat/multilingual-e5-base")
index = faiss.read_index("data/index/index.faiss")
metadata = []
with open("data/index/metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

# Get real queries from the dataset
rows = [json.loads(l) for l in open("data/processed/passages.jsonl", encoding="utf-8")]
import random
random.seed(42)
hi_rows = [r for r in rows if r["lang"] == "hi"]
samples = random.sample(hi_rows, 5)

print("=== Testing with real Hindi queries from dataset ===\n")
for r in samples:
    q = r["query"]
    vec = model.encode([q], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
    scores, ids = index.search(vec, 2)
    print(f"Query: {q[:50]}")
    for s, i in zip(scores[0], ids[0]):
        m = metadata[i]
        print(f"  score={s:.4f} lang={m['lang']} text={m['text'][:70]}")
    print()