"""
Stored-vector check: does the ONNX e5 encoder reproduce the vectors already in the index?

Embeds random passages from metadata.jsonl with backend.encoders.E5Encoder (the exact code and image the
server uses) and compares each with its stored vector in vectors_fp16.bin. Those vectors were made by the
original PyTorch model, so the index itself is the reference — no PyTorch needed.

    MODAL_PROFILE=rawrmeinkayanosaurushun modal run eval/check_embeddings.py

Pass bar (Phase 3): average cosine >= 0.999 and lowest >= 0.995.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import modal

from modal_app import image, volume

app = modal.App("voice-rag-check-embeddings", image=image)


@app.function(gpu="A100", volumes={"/index": volume}, timeout=1800)
def check(n: int = 1000, seed: int = 7) -> dict:
    import collections
    import json
    import random
    import struct

    import numpy as np
    import onnxruntime as ort

    ort.preload_dlls()
    from backend.encoders import E5Encoder

    encoder = E5Encoder("/models/e5", max_length=64)
    n_rows = os.path.getsize("/index/metadata.offsets") // 8
    vectors = np.memmap("/index/vectors_fp16.bin", dtype=np.float16, mode="r", shape=(n_rows, 768))
    rng = random.Random(seed)
    results = []
    with open("/index/metadata.offsets", "rb") as fo, open("/index/metadata.jsonl", "rb") as fm:
        for _ in range(n):
            i = rng.randrange(n_rows)
            fo.seek(i * 8)
            fm.seek(struct.unpack("<q", fo.read(8))[0])
            row = json.loads(fm.readline())
            stored = vectors[i].astype(np.float32)
            stored /= np.linalg.norm(stored)
            cosine = float(encoder.embed(row["text"]) @ stored)
            results.append((cosine, row.get("lang"), row.get("chunk_strategy"), row["text"][:80]))

    cos = np.array([r[0] for r in results])
    by = collections.defaultdict(list)
    for c, lang, strategy, _ in results:
        by[f"lang={lang}"].append(c)
        by[f"strategy={strategy}"].append(c)
    worst = sorted(results)[:5]
    return {
        "n": n, "mean": float(cos.mean()), "min": float(cos.min()),
        "p1": float(np.percentile(cos, 1)), "below_0.999": int((cos < 0.999).sum()), "below_0.995": int((cos < 0.995).sum()),
        "groups": {k: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v))} for k, v in sorted(by.items())},
        "worst": [{"cosine": c, "lang": lang, "strategy": s, "text": t} for c, lang, s, t in worst],
        "passed": bool(cos.mean() >= 0.999 and cos.min() >= 0.995),
    }


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(check.remote(), ensure_ascii=False, indent=1))
