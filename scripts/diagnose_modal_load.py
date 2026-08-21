import modal
import os

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-diagnostic", image=modal.Image.debian_slim().pip_install("faiss-cpu", "numpy", "torch", "transformers", "sentence-transformers"))

@app.cls(gpu="A10G", memory=32768, volumes={"/index": volume})
class DiagnosticLoader:
    @modal.method()
    def test_load(self):
        import time, torch, faiss, numpy as np
        print("Checking index files in /index volume:")
        for f in os.listdir("/index"):
            sz = os.path.getsize(f"/index/{f}")
            print(f"  - {f}: {sz / (1024**3):.2f} GB ({sz:,} bytes)")

        print("\n1. Testing MMAP FAISS read...")
        t0 = time.perf_counter()
        idx = faiss.read_index("/index/index.faiss", faiss.IO_FLAG_MMAP)
        print(f"  FAISS MMAP read {idx.ntotal:,} vectors in {time.perf_counter()-t0:.2f}s")

        print("\n2. Testing vector array extraction...")
        t0 = time.perf_counter()
        raw_bytes = faiss.vector_to_array(idx.codes)
        print(f"  vector_to_array done in {time.perf_counter()-t0:.2f}s")
        raw_vecs = raw_bytes.view(np.float32).reshape(idx.ntotal, idx.d)

        print("\n3. Testing GPU VRAM transfer...")
        t0 = time.perf_counter()
        gpu_vecs = torch.empty((idx.ntotal, idx.d), dtype=torch.float16, device="cuda")
        chunk_sz = 2_000_000
        for c_start in range(0, idx.ntotal, chunk_sz):
            c_end = min(c_start + chunk_sz, idx.ntotal)
            gpu_vecs[c_start:c_end] = torch.from_numpy(raw_vecs[c_start:c_end]).half().cuda()
        print(f"  GPU transfer done in {time.perf_counter()-t0:.2f}s ({gpu_vecs.element_size()*gpu_vecs.nelement()/(1024**3):.2f} GB in VRAM)")

        print("\n4. Testing metadata offset table...")
        offsets_path = "/index/metadata.offsets"
        t0 = time.perf_counter()
        if os.path.exists(offsets_path):
            offsets = np.fromfile(offsets_path, dtype=np.int64)
            print(f"  Offsets loaded from cache in {time.perf_counter()-t0:.2f}s ({len(offsets):,} entries)")
        else:
            print("  Offsets file not found, building and saving now...")
            offsets_list = [0]
            with open("/index/metadata.jsonl", "rb") as f:
                pos = 0
                buf_size = 1024 * 1024 * 8
                while True:
                    chunk = f.read(buf_size)
                    if not chunk: break
                    nl_pos = 0
                    while True:
                        i = chunk.find(b"\n", nl_pos)
                        if i == -1: break
                        offsets_list.append(pos + i + 1)
                        nl_pos = i + 1
                    pos += len(chunk)
            if offsets_list and offsets_list[-1] >= pos: offsets_list.pop()
            offsets = np.array(offsets_list, dtype=np.int64)
            offsets.tofile(offsets_path)
            volume.commit()
            print(f"  Offsets generated and saved in {time.perf_counter()-t0:.2f}s ({len(offsets):,} entries)")

        return "SUCCESS"

@app.local_entrypoint()
def main():
    d = DiagnosticLoader()
    res = d.test_load.remote()
    print("Diagnostic Result:", res)
