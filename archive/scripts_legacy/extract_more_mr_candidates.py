import modal
import json

app = modal.App("voice-rag-candidate-extractor-set2")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("voice-rag-index")

@app.function(image=image, volumes={"/index": volume}, timeout=300)
def extract_more_mr(exclude_qids: list):
    exclude_set = set(str(x) for x in exclude_qids)
    candidates = []
    
    with open("/index/metadata.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lang = item.get("lang")
            qid = str(item.get("query_id", ""))
            q = item.get("query", "").strip()
            ans = item.get("answer", "").strip()
            text = item.get("text", "").strip()
            
            if lang == "mr" and q and ans and qid not in exclude_set:
                if len(q) > 8 and len(ans) > 3:
                    if ans.lower() in text.lower() or len(ans.split()) <= 6:
                        candidates.append({
                            "id": f"mr_{len(candidates)+1}",
                            "query_id": qid,
                            "query": q,
                            "ground_truth_answer": ans,
                        })
                        if len(candidates) >= 200:
                            break
                        
    return candidates

@app.local_entrypoint()
def main():
    with open("data/benchmark_90_verified.json", "r", encoding="utf-8") as f:
        set1 = json.load(f)
    with open("data/benchmark_90_set2_verified.json", "r", encoding="utf-8") as f:
        set2 = json.load(f)
        
    exclude_qids = []
    for d in [set1, set2]:
        for lang, items in d.items():
            for item in items:
                exclude_qids.append(item.get("query_id"))
                
    print(f"Requesting new Marathi candidates from Modal volume excluding {len(exclude_qids)} QIDs...")
    new_mr = extract_more_mr.remote(exclude_qids)
    print(f"Extracted {len(new_mr)} new Marathi candidates.")
    
    with open("data/raw_mr_set2_pool.json", "w", encoding="utf-8") as f:
        json.dump(new_mr, f, indent=2, ensure_ascii=False)
    print("Saved to data/raw_mr_set2_pool.json")

if __name__ == "__main__":
    main()
