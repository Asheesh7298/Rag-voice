import json
import urllib.request
import urllib.parse

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

with open("data/benchmark_90_verified.json", "r", encoding="utf-8") as f:
    s1 = json.load(f)
with open("data/benchmark_90_set2_verified.json", "r", encoding="utf-8") as f:
    s2 = json.load(f)

def test_query(q):
    data = urllib.parse.urlencode({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("Evaluating clean questions from datasets...")
clean_matches = []

for s_name, ds in [("Set1", s1), ("Set2", s2)]:
    for lang in ["en", "hi", "mr"]:
        for item in ds[lang]:
            q = item["query"].strip()
            gt = item["ground_truth_answer"].strip()
            
            # Filter for concise, very clear factual questions
            if len(q.split()) <= 8 and len(gt.split()) <= 12:
                try:
                    res = test_query(q)
                    ans = res.get("answer", "").strip()
                    guard = res.get("guardrail_triggered")
                    conf = res.get("confidence", 0.0)
                    
                    if not guard and ans and conf > 0.4:
                        # Check if ground truth or answer strongly overlap
                        gt_low = gt.lower()
                        ans_low = ans.lower()
                        if any(w in ans_low for w in gt_low.split() if len(w) > 3) or ans_low in gt_low or gt_low in ans_low:
                            clean_matches.append({
                                "lang": lang,
                                "query": q,
                                "answer": ans,
                                "ground_truth": gt,
                                "conf": conf,
                                "source": s_name
                            })
                            print(f"[{lang.upper()}] Q: {q}")
                            print(f"       Ans: {ans}")
                            print(f"       GT:  {gt}")
                            print(f"       Conf: {conf:.2f}")
                            print("-" * 50)
                            if len(clean_matches) >= 20:
                                break
                except Exception as e:
                    pass
        if len(clean_matches) >= 20:
            break

with open("data/clean_example_candidates.json", "w", encoding="utf-8") as f:
    json.dump(clean_matches, f, indent=2, ensure_ascii=False)

print("Saved clean candidates!")
