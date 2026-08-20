import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

terms = ["new delhi", "delhi", "नई दिल्ली", "नवी दिल्ली", "paris", "पॅरिस", "पेरिस", "france", "capital of india", "capital of france"]
found = {t: [] for t in terms}

with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        p = json.loads(line)
        text = (p.get('text', '') + " " + p.get('query', '')).lower()
        for t in terms:
            if t in text:
                found[t].append(p)

for t, hits in found.items():
    print(f"Term: '{t}' -> Found: {len(hits)} passages")
    for h in hits[:2]:
        print(f"  [{h['lang']}] Query: '{h.get('query')}' | Text: '{h.get('text')[:100]}'")
