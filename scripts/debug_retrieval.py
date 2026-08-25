import modal

app = modal.App("debug-query")
vol = modal.Volume.from_name("voice-rag-index")
image = modal.Image.debian_slim().pip_install("torch", "sentence-transformers", "transformers", "numpy", "faiss-cpu")

@app.function(volumes={"/index": vol}, gpu="A10G", timeout=120)
def debug(q="how much does it cost to change a jeep alternator"):
    import torch, numpy as np, os, json
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    m = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
    qv = m.encode([f"query: {q}"], normalize_embeddings=True)
    
    vecs = np.memmap("/index/vectors_fp16.bin", dtype=np.float16, mode="r", shape=(13020220, 768))
    t_vecs = torch.from_numpy(vecs).cuda()
    qv_t = torch.from_numpy(qv).cuda().half()
    sims = torch.matmul(t_vecs, qv_t.T).squeeze(1)
    top_scores, top_ids = torch.topk(sims, k=10)
    
    offsets = np.fromfile("/index/metadata.offsets", dtype=np.int64)
    with open("/index/metadata.jsonl", "rb") as f:
        for sc, idx in zip(top_scores.cpu().numpy(), top_ids.cpu().numpy()):
            f.seek(int(offsets[idx]))
            meta = json.loads(f.readline().decode("utf-8"))
            print(f"\n[Score: {sc:.4f} | Lang: {meta.get('lang')} | ID: {meta.get('query_id')}]")
            print(f"Text: {meta.get('text')[:200]}...")

@app.local_entrypoint()
def main():
    debug.remote()
