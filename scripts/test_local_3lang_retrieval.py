import os
import sys
import time
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "index_3lang.faiss")
META_PATH = os.path.join(BASE_DIR, "data", "index", "metadata_3lang.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading FAISS index on {device}...")
index = faiss.read_index(INDEX_PATH)
print(f"Index loaded. Total vectors: {index.ntotal}")

metadata = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def search_query(query: str, top_k=3):
    t0 = time.perf_counter()
    inputs = tokenizer([f"query: {query}"], max_length=256, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy().astype("float32")
    t_embed = round((time.perf_counter() - t0) * 1000, 2)

    t1 = time.perf_counter()
    scores, indices = index.search(emb, top_k)
    t_search = round((time.perf_counter() - t1) * 1000, 2)
    t_total = round((time.perf_counter() - t0) * 1000, 2)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            m = metadata[idx]
            results.append({
                "score": round(float(score), 4),
                "lang": m.get("lang"),
                "text": m.get("text")[:120] + "..."
            })
    return results, t_embed, t_search, t_total

test_queries = [
    "एनसीआईएस का निदेशक कौन है?",
    "क्रोमॅटोग्राफी म्हणजे काय",
    "How fast does a jetliner fly?",
    "एक छोटे व्यवसाय के लिए एक अकाउंटेंट की क्या आवश्यकता है?",
    "विद्यार्थी निवासाचा खर्च किती आहे",
]

print("\n" + "=" * 80)
print("TESTING LOCAL RTX 4050 GPU RETRIEVAL PERFORMANCE")
print("=" * 80)

for q in test_queries:
    res, t_emb, t_s, t_tot = search_query(q)
    print(f"\nQuery: {q}")
    print(f"  Embedding: {t_emb} ms | FAISS Search: {t_s} ms | Total: {t_tot} ms")
    if res:
        top = res[0]
        print(f"  Top Match [{top['lang']}] (Score: {top['score']}): {top['text']}")
