import json
import urllib.request
import urllib.parse
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('data/sample_40_eval.json', 'r', encoding='utf-8') as f:
    sample_40 = json.load(f)[:40]

ENDPOINT = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run/query"

print("Detailed Diagnosis of All 40 Benchmark Queries:\n")

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
        res = {'answer': str(e), 'sources': [], 'guardrail_triggered': 'error'}

    ans = res.get('answer', '')
    guard = res.get('guardrail_triggered')
    sources = res.get('sources', [])
    top_source = sources[0]['text'] if sources else "NO SOURCE"

    print(f"[{i+1}] ({lang}) Q: {q}")
    print(f"    Gold:      {gold}")
    print(f"    Extracted: {ans}")
    print(f"    Guardrail: {guard}")
    print(f"    Top passage snippet: {top_source[:160]}...")
    print("-" * 80)
