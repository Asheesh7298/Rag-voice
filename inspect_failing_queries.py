import json
import urllib.request
import urllib.parse
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/sample_40_eval.json', 'r', encoding='utf-8') as f:
    sample_40 = json.load(f)[:40]

ENDPOINT = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run/query"

# Load passages to see exact ground-truth passage text for each query
passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

q_to_passage = {p['query'].strip().lower(): p for p in passages if p.get('query')}

print("Analyzing the 10 failing questions in detail:")
for i, item in enumerate(sample_40):
    q = item['query']
    lang = item['lang']
    gold = item['gold_answer']

    data = urllib.parse.urlencode({'query': q}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        res = {'answer': str(e), 'sources': []}

    ans = res.get('answer', '')
    sources = res.get('sources', [])
    gt_p = q_to_passage.get(q.strip().lower())
    gt_text = gt_p.get('text', '') if gt_p else 'N/A'
    gt_pid = gt_p.get('id', '') if gt_p else 'N/A'

    # Check if this was one of the 10 incorrect ones
    # Check if ground truth passage was in sources
    found_gt_in_sources = False
    gt_rank = -1
    for s_idx, s in enumerate(sources):
        if gt_text and (gt_text[:60] in s.get('text', '') or s.get('text', '')[:60] in gt_text):
            found_gt_in_sources = True
            gt_rank = s_idx + 1
            break

    print(f"\n--- [Q{i+1}] ({lang}) {q} ---")
    print(f"  Gold Answer:    {gold}")
    print(f"  Model Answer:   {ans}")
    print(f"  GT passage in top-{len(sources)} sources? {found_gt_in_sources} (Rank: {gt_rank})")
    if not found_gt_in_sources:
        print(f"  GT Passage ID: {gt_pid}")
        print(f"  GT Passage: {gt_text[:120]}...")
    if sources:
        print(f"  Top source: {sources[0]['text'][:120]}...")
