import modal
import os

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-fp16-converter", image=modal.Image.debian_slim().pip_install("faiss-cpu", "numpy", "torch"))

@app.cls(gpu="A10G", memory=65536, volumes={"/index": volume})
class FP16Converter:
    @modal.method()
    def convert(self):
        import time, faiss, numpy as np, torch
        print("1. Reading FAISS index via MMAP...")
        t0 = time.perf_counter()
        idx = faiss.read_index("/index/index.faiss", faiss.IO_FLAG_MMAP)
        print(f"   Read {idx.ntotal:,} vectors in {time.perf_counter()-t0:.2f}s")

        print("2. Extracting raw float32 array...")
        t0 = time.perf_counter()
        raw_bytes = faiss.vector_to_array(idx.codes)
        raw_vecs = raw_bytes.view(np.float32).reshape(idx.ntotal, idx.d)
        print(f"   Extracted in {time.perf_counter()-t0:.2f}s")

        print("3. Converting and writing directly to /index/vectors_fp16.bin...")
        t0 = time.perf_counter()
        fp16_path = "/index/vectors_fp16.bin"
        fp16_mmap = np.memmap(fp16_path, dtype=np.float16, mode="w+", shape=raw_vecs.shape)
        
        chunk_sz = 1_000_000
        for i in range(0, idx.ntotal, chunk_sz):
            end = min(i + chunk_sz, idx.ntotal)
            fp16_mmap[i:end] = raw_vecs[i:end].astype(np.float16)
        fp16_mmap.flush()
        print(f"   Saved {os.path.getsize(fp16_path)/(1024**3):.2f} GB in {time.perf_counter()-t0:.2f}s")

        print("4. Committing Modal Volume...")
        volume.commit()
        print("✅ Modal Volume committed with vectors_fp16.bin!")
        return "CONVERT_SUCCESS"

@app.local_entrypoint()
def main():
    c = FP16Converter()
    res = c.convert.remote()
    print("Converter Result:", res)
