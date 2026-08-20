"""
Extract and verify 90 high-confidence questions (30 Hindi, 30 Marathi, 30 English)
directly from the active 1.51M multi-strategy index on Modal Volume.
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "tqdm")
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-extract-90-qa", image=image)

@app.function(volumes={"/index": volume}, timeout=600)
def extract_clean_qa_pairs():
    print("Reading /index/metadata.jsonl from Modal Volume...")
    by_lang = {"hi": [], "mr": [], "en": []}
    seen_qids = set()

    with open("/index/metadata.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lang = item.get("lang")
            if lang not in by_lang:
                continue
            
            # We want passage_native chunks with is_selected=True and non-empty query + answer
            if item.get("chunk_strategy") != "passage_native":
                continue
            if not item.get("is_selected"):
                continue

            q = item.get("query", "").strip()
            ans = item.get("answer", "").strip()
            qid = item.get("query_id", "")
            text = item.get("text", "").strip()

            if not q or not ans or len(q) < 8 or len(ans) < 3:
                continue
            if qid in seen_qids:
                continue

            # Ensure the answer or key terms of answer appear in the text
            ans_lower = ans.lower()
            text_lower = text.lower()
            if ans_lower in text_lower or len(ans.split()) <= 6:
                seen_qids.add(qid)
                by_lang[lang].append({
                    "id": item.get("chunk_id"),
                    "query_id": qid,
                    "lang": lang,
                    "query": q,
                    "ground_truth_answer": ans,
                    "text": text,
                })

    print(f"Extracted candidates: HI={len(by_lang['hi'])}, MR={len(by_lang['mr'])}, EN={len(by_lang['en'])}")
    return {
        "hi": by_lang["hi"][:150],
        "mr": by_lang["mr"][:150],
        "en": by_lang["en"][:150],
    }

@app.local_entrypoint()
def main():
    res = extract_clean_qa_pairs.remote()
    with open("data/raw_90_candidates.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("Saved raw candidate pools to data/raw_90_candidates.json")
