import modal
import os

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "sentence-transformers>=3.0.0",
        "transformers>=4.44.0",
        "torch>=2.4.0",
        "faiss-cpu>=1.8.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.26.0",
        "httpx>=0.27.0",
        "python-multipart>=0.0.9",
        "accelerate>=0.33.0",
        "orjson>=3.10.0",
    )
)

app = modal.App("voice-rag-trace", image=image)

@app.cls(gpu="A10G", memory=32768, volumes={"/index": volume})
class TraceRunner:
    @modal.method()
    def run_trace(self):
        import time, torch, numpy as np
        print("[TRACE] 1. Reloading volume...")
        t0 = time.perf_counter()
        volume.reload()
        print(f"[TRACE] Volume reloaded in {time.perf_counter()-t0:.2f}s")

        print("[TRACE] 2. Checking /index/vectors_fp16.bin...")
        fp16_bin = "/index/vectors_fp16.bin"
        if os.path.exists(fp16_bin):
            sz = os.path.getsize(fp16_bin)
            print(f"[TRACE] vectors_fp16.bin exists: {sz/(1024**3):.2f} GB")
        else:
            print("[TRACE] vectors_fp16.bin NOT FOUND!")

        print("[TRACE] 3. Loading vectors_fp16.bin directly into A10G GPU VRAM...")
        t0 = time.perf_counter()
        mmap_vecs = np.memmap(fp16_bin, dtype=np.float16, mode="r", shape=(13020220, 768))
        gpu_vecs = torch.empty((13020220, 768), dtype=torch.float16, device="cuda")
        chunk_sz = 2_000_000
        for c_start in range(0, 13020220, chunk_sz):
            c_end = min(c_start + chunk_sz, 13020220)
            gpu_vecs[c_start:c_end] = torch.from_numpy(mmap_vecs[c_start:c_end]).cuda()
        print(f"[TRACE] GPU transfer complete in {time.perf_counter()-t0:.2f}s!")

        print("[TRACE] 4. Loading metadata offsets...")
        t0 = time.perf_counter()
        offsets = np.fromfile("/index/metadata.offsets", dtype=np.int64)
        print(f"[TRACE] Offsets loaded ({len(offsets):,} entries) in {time.perf_counter()-t0:.2f}s!")

        print("[TRACE] 5. Running test search with dummy vector...")
        t0 = time.perf_counter()
        dummy_q = torch.randn(1, 768, dtype=torch.float16, device="cuda")
        sims = torch.matmul(gpu_vecs, dummy_q.T).squeeze(1)
        top_scores, top_ids = torch.topk(sims, k=10)
        print(f"[TRACE] 13M GPU Search done in {(time.perf_counter()-t0)*1000:.2f} ms! Top score: {top_scores[0].item():.4f}")

        print("[TRACE] 6. Testing random metadata seek & parse...")
        t0 = time.perf_counter()
        import orjson
        with open("/index/metadata.jsonl", "rb") as f:
            for idx in top_ids.cpu().numpy().tolist():
                f.seek(int(offsets[idx]))
                rec = orjson.loads(f.readline())
        print(f"[TRACE] 10 Metadata seeks & json decode done in {(time.perf_counter()-t0)*1000:.2f} ms!")

        return "TRACE_COMPLETE_SUCCESS"

@app.local_entrypoint()
def main():
    tr = TraceRunner()
    res = tr.run_trace.remote()
    print("Result:", res)
