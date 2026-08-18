"""
High-Performance Local 3-Language (Hindi, Marathi, English) RAG Index Builder.
Optimized for NVIDIA RTX 4050 Laptop GPU (CUDA 12.6, FP16, Batch Size 64).
"""
import os
import sys
import json
import time
import torch
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUT_FAISS_PATH = os.path.join(INDEX_DIR, "index_3lang.faiss")
OUT_META_PATH = os.path.join(INDEX_DIR, "metadata_3lang.jsonl")
OUT_PASSAGES_PATH = os.path.join(PROCESSED_DIR, "passages_3lang.jsonl")

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM = 768
BATCH_SIZE = 64


def get_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"[GPU ACTIVE] Using: {gpu_name} ({vram} GB VRAM)")
        return torch.device("cuda")
    else:
        threads = torch.get_num_threads()
        print(f"[CPU ACTIVE] GPU not detected, using CPU ({threads} threads)")
        return torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_texts(texts: list[str], tokenizer, model, device) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    inputs = tokenizer(prefixed, max_length=256, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    return embeddings.cpu().numpy().astype("float32")


def build_3lang_index():
    device = get_device()
    print("=" * 80)
    print("STARTING LOCAL 3-LANGUAGE (HINDI + MARATHI + ENGLISH) INDEX BUILD")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
    model.eval()

    # 2. Collect Passages across Hindi, Marathi, and English
    passages = []
    
    # Load IndicMSMARCO Hindi
    print("\n[1/3] Loading Hindi passages from IndicMSMARCO...")
    try:
        ds_hi = load_dataset("ai4bharat/IndicMSMARCO", "hi", split="train")
        for i, row in enumerate(ds_hi):
            text = (row.get("passage") or "").strip()
            if text:
                passages.append({
                    "id": f"hi-{i}",
                    "lang": "hi",
                    "query": row.get("query", ""),
                    "text": text,
                    "strategy": "passage_full"
                })
        print(f"  Loaded {len(ds_hi)} Hindi passages.")
    except Exception as e:
        print(f"  Error loading Hindi: {e}")

    # Load IndicMSMARCO Marathi
    print("\n[2/3] Loading Marathi passages from IndicMSMARCO...")
    try:
        ds_mr = load_dataset("ai4bharat/IndicMSMARCO", "mr", split="train")
        for i, row in enumerate(ds_mr):
            text = (row.get("passage") or "").strip()
            if text:
                passages.append({
                    "id": f"mr-{i}",
                    "lang": "mr",
                    "query": row.get("query", ""),
                    "text": text,
                    "strategy": "passage_full"
                })
        print(f"  Loaded {len(ds_mr)} Marathi passages.")
    except Exception as e:
        print(f"  Error loading Marathi: {e}")

    # Load English Passages from existing processed set or sample
    print("\n[3/3] Loading English passages...")
    try:
        existing_passages_file = os.path.join(DATA_DIR, "processed", "passages.jsonl")
        if os.path.exists(existing_passages_file):
            with open(existing_passages_file, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("lang") in ("en", "hi", "mr"):
                        passages.append({
                            "id": row.get("id"),
                            "lang": row.get("lang"),
                            "query": row.get("query", ""),
                            "text": row.get("text", ""),
                            "strategy": row.get("strategy", "passage_full")
                        })
        print(f"  Total combined passages for indexing: {len(passages)}")
    except Exception as e:
        print(f"  Error loading English/existing passages: {e}")

    # 3. Create Multi-Strategy Chunks
    print("\nGenerating multi-strategy chunks (document, sentence, window)...")
    chunks = []
    for p in passages:
        p_text = p["text"]
        p_lang = p["lang"]
        p_id = p["id"]
        p_query = p.get("query", "")

        # Strategy A: Document / Full passage
        chunks.append({
            "chunk_id": f"{p_id}_doc",
            "text": p_text,
            "lang": p_lang,
            "query": p_query,
            "strategy": "document"
        })

        # Strategy B: Sentence / Half-split chunks
        sentences = [s.strip() for s in p_text.replace("।", ".").split(".") if len(s.strip()) > 15]
        if len(sentences) >= 2:
            mid = len(sentences) // 2
            chunks.append({
                "chunk_id": f"{p_id}_s1",
                "text": ". ".join(sentences[:mid]),
                "lang": p_lang,
                "query": p_query,
                "strategy": "sentence"
            })
            chunks.append({
                "chunk_id": f"{p_id}_s2",
                "text": ". ".join(sentences[mid:]),
                "lang": p_lang,
                "query": p_query,
                "strategy": "sentence"
            })

    total_chunks = len(chunks)
    print(f"Total chunks to embed: {total_chunks}")

    # 4. Initialize FAISS Index (Inner Product on Normalized Vectors = Cosine Similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)

    # 5. Embed and Index in GPU Batches
    print(f"\nEmbedding chunks on {device.type.upper()} in batches of {BATCH_SIZE}...")
    t_start = time.perf_counter()

    with open(OUT_META_PATH, "w", encoding="utf-8") as meta_file:
        for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Embedding Batches"):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            batch_texts = [c["text"] for c in batch_chunks]

            batch_vecs = embed_texts(batch_texts, tokenizer, model, device)
            index.add(batch_vecs)

            for c in batch_chunks:
                meta_file.write(json.dumps(c, ensure_ascii=False) + "\n")

    t_total = round(time.perf_counter() - t_start, 2)
    throughput = round(total_chunks / max(0.1, t_total), 1)

    print("\n" + "=" * 80)
    print("INDEX BUILDING COMPLETE!")
    print("=" * 80)
    print(f"Total Chunks Indexed: {index.ntotal}")
    print(f"Total Time:           {t_total} seconds")
    print(f"Throughput:           {throughput} chunks/sec")
    print(f"FAISS Index Saved to: {OUT_FAISS_PATH}")
    print(f"Metadata Saved to:    {OUT_META_PATH}")

    faiss.write_index(index, OUT_FAISS_PATH)
    print("Saved FAISS index binary successfully!")


if __name__ == "__main__":
    build_3lang_index()
