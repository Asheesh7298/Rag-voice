import modal
import sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0.0",
        "transformers>=4.44.0",
        "torch>=2.4.0",
        "accelerate>=0.33.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "numpy>=1.26.0",
        "orjson>=3.10.0",
    )
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base').save('/models/e5-base')\"",
        "python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').save_pretrained('/models/qwen-0.5b'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='auto').save_pretrained('/models/qwen-0.5b')\"",
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-3tests", image=image)

@app.cls(gpu="A10G", memory=32768, timeout=300, volumes={"/index": volume})
class LiveTester:
    @modal.enter()
    def setup(self):
        import time, torch, os, numpy as np, orjson
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("[SETUP] 1. Loading embedding model...")
        self.embed_model = SentenceTransformer("/models/e5-base", device="cuda")
        self.embed_model.max_seq_length = 64
        self.embed_model.encode(["warmup"], normalize_embeddings=True)

        print("[SETUP] 2. Loading 13.02M vectors into GPU VRAM...")
        fp16_bin = "/index/vectors_fp16.bin"
        mmap_vecs = np.memmap(fp16_bin, dtype=np.float16, mode="r", shape=(13020220, 768))
        self.gpu_vectors = torch.empty((13020220, 768), dtype=torch.float16, device="cuda")
        chunk_sz = 2_000_000
        for c_start in range(0, 13020220, chunk_sz):
            c_end = min(c_start + chunk_sz, 13020220)
            self.gpu_vectors[c_start:c_end] = torch.from_numpy(mmap_vecs[c_start:c_end]).cuda()

        print("[SETUP] 3. Loading metadata offset table & file...")
        self.offsets = np.fromfile("/index/metadata.offsets", dtype=np.int64)
        self.meta_file = open("/index/metadata.jsonl", "rb")

        print("[SETUP] 4. Loading Qwen2.5-0.5B-Instruct Mini-LLM...")
        self.qwen_tok = AutoTokenizer.from_pretrained("/models/qwen-0.5b")
        self.qwen_mod = AutoModelForCausalLM.from_pretrained("/models/qwen-0.5b", torch_dtype=torch.float16, device_map="cuda")
        self.qwen_mod.eval()
        # Warmup
        _w = self.qwen_tok(["hi"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            self.qwen_mod.generate(**_w, max_new_tokens=2)
        print("✅ Ready for queries!")

    @modal.method()
    def query(self, text: str, lang_code: str):
        import time, torch, orjson, numpy as np
        t_start = time.perf_counter()
        
        # 1. Embed query
        t0 = time.perf_counter()
        qv = self.embed_model.encode([text], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
        norm = np.linalg.norm(qv)
        if norm > 0: qv = qv / norm
        embed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Dense GPU Vector Search
        t0 = time.perf_counter()
        qv_t = torch.from_numpy(qv).cuda().half()
        sims = torch.matmul(self.gpu_vectors, qv_t.T).squeeze(1)
        top_scores, top_ids = torch.topk(sims, k=15)
        search_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Retrieve metadata
        t0 = time.perf_counter()
        chunks = []
        for s, idx in zip(top_scores.cpu().tolist(), top_ids.cpu().tolist()):
            self.meta_file.seek(int(self.offsets[idx]))
            rec = orjson.loads(self.meta_file.readline())
            rec["score"] = float(s)
            chunks.append(rec)
        meta_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Filter by language if matching
        filtered = [c for c in chunks if c.get("lang") == lang_code]
        use_chunks = filtered[:3] if filtered else chunks[:3]

        # 4. Generate grounded answer with Qwen2.5-0.5B
        t0 = time.perf_counter()
        facts = "\n".join([f"- {c['text'].strip()}" for c in use_chunks if c.get("text")])
        lang_names = {"hi": "Hindi (हिंदी)", "mr": "Marathi (मराठी)", "en": "English"}
        tgt_lang = lang_names.get(lang_code, "English")

        sys_prompt = f"You are VoxLore, a strict factual voice assistant. Answer in under 20 words in {tgt_lang} using ONLY the facts provided. If facts do not contain the answer, say 'I do not have sufficient information.' Do NOT invent facts."
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Facts:\n{facts}\n\nQuestion: {text}\nAnswer:"}
        ]
        text_in = self.qwen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_tok([text_in], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.qwen_mod.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.qwen_tok.eos_token_id,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.qwen_tok.decode(gen_tokens, skip_special_tokens=True).strip()
        gen_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "query": text,
            "answer": answer,
            "top_fact": use_chunks[0]["text"][:120] + "..." if use_chunks else "",
            "timings": {
                "embed_ms": embed_ms,
                "gpu_search_13m_ms": search_ms,
                "meta_seek_ms": meta_ms,
                "qwen_gen_ms": gen_ms,
                "total_ms": total_ms,
            }
        }

@app.local_entrypoint()
def main():
    tester = LiveTester()
    
    tests = [
        ("what county is columbus city in", "en", "ENGLISH"),
        ("ब्राइटन टाउनशिप फोन नंबर", "hi", "HINDI"),
        ("फ्रान्सचे सध्याचे चलन काय आहे", "mr", "MARATHI"),
    ]
    
    print("\n" + "=" * 75)
    print("      LIVE 13.02M MULTI-STRATEGY + QWEN2.5-0.5B TEST RUN (3 LANGUAGES)")
    print("=" * 75)
    
    for q, lang, name in tests:
        print(f"\n[{name} QUERY] \"{q}\"")
        res = tester.query.remote(q, lang)
        print(f"  💬 Answer:     {res['answer']}")
        print(f"  📖 Top Source: {res['top_fact']}")
        t = res['timings']
        print(f"  ⚡ Timings:    Total={t['total_ms']}ms | GPU_Search(13M)={t['gpu_search_13m_ms']}ms | Qwen_Gen={t['qwen_gen_ms']}ms | Embed={t['embed_ms']}ms")
    
    print("\n" + "=" * 75)
    print("                             ALL 3 TESTS COMPLETE")
    print("=" * 75)
