import modal
import os

image = modal.Image.debian_slim().pip_install("faiss-cpu", "numpy", "torch")
volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-postprocess-index", image=image)

@app.cls(gpu="A10G", memory=65536, volumes={"/index": volume}, timeout=1200)
class IndexPostProcessor:

    @modal.method()
    def process(self):
        import time, faiss, numpy as np, torch

        # ── 1. Convert index.faiss to vectors_fp16.bin ──
        fp16_path = "/index/vectors_fp16.bin"
        if not os.path.exists(fp16_path):
            print("1. Reading FAISS index via MMAP...")
            t0 = time.perf_counter()
            idx = faiss.read_index("/index/index.faiss", faiss.IO_FLAG_MMAP)
            print(f"   Read {idx.ntotal:,} vectors in {time.perf_counter()-t0:.2f}s")

            print("2. Extracting raw float32 array...")
            t0 = time.perf_counter()
            raw_bytes = faiss.vector_to_array(idx.codes)
            raw_vecs = raw_bytes.view(np.float32).reshape(idx.ntotal, idx.d)
            print(f"   Extracted {raw_vecs.shape} in {time.perf_counter()-t0:.2f}s")

            print("3. Converting and writing to /index/vectors_fp16.bin...")
            t0 = time.perf_counter()
            fp16_mmap = np.memmap(fp16_path, dtype=np.float16, mode="w+", shape=raw_vecs.shape)
            
            chunk_sz = 1_000_000
            for i in range(0, idx.ntotal, chunk_sz):
                end = min(i + chunk_sz, idx.ntotal)
                fp16_mmap[i:end] = raw_vecs[i:end].astype(np.float16)
            fp16_mmap.flush()
            print(f"   ✅ Saved {os.path.getsize(fp16_path)/(1024**3):.2f} GB in {time.perf_counter()-t0:.2f}s")
        else:
            print("✅ /index/vectors_fp16.bin already exists!")

        # ── 2. Build metadata.offsets ──
        offsets_path = "/index/metadata.offsets"
        if not os.path.exists(offsets_path):
            print("\n4. Building metadata binary offsets...")
            t0 = time.perf_counter()
            offsets = []
            with open("/index/metadata.jsonl", "rb") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(pos)
                    if len(offsets) % 2_000_000 == 0:
                        print(f"   Indexed {len(offsets):,} lines ({time.perf_counter()-t0:.1f}s)...")
            
            arr = np.array(offsets, dtype=np.int64)
            arr.tofile(offsets_path)
            np.save("/index/metadata_offsets.npy", arr)
            print(f"   ✅ Saved {len(arr):,} offsets in {time.perf_counter()-t0:.1f}s ({os.path.getsize(offsets_path)/(1024*1024):.1f} MB)")
        else:
            print("✅ /index/metadata.offsets already exists!")

        print("\n5. Committing Modal Volume...")
        volume.commit()
        print("🎉 Modal Volume successfully optimized and committed!")
        return "SUCCESS"

@app.local_entrypoint()
def main():
    proc = IndexPostProcessor()
    res = proc.process.remote()
    print("Optimization Result:", res)
