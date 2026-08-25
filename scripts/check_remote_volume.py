import os
import modal

image = modal.Image.debian_slim().pip_install("faiss-cpu", "numpy")
vol = modal.Volume.from_name("voice-rag-index")
app = modal.App("check-volume-status", image=image)

@app.function(volumes={"/index": vol}, timeout=120)
def check():
    files = os.listdir("/index")
    print("Files in /index volume:")
    for f in sorted(files):
        p = os.path.join("/index", f)
        sz = os.path.getsize(p)
        print(f"  • {f}: {sz / (1024**2):.2f} MB ({sz / (1024**3):.2f} GB)")
    if "index.faiss" in files:
        import faiss
        idx = faiss.read_index("/index/index.faiss")
        print(f"\n✅ FAISS index loaded! Total vectors: {idx.ntotal:,}")
    return files

@app.local_entrypoint()
def main():
    check.remote()
