import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

queries_to_check = [
    "भारत की राजधानी",
    "राजधानी",
    "photosynthesis",
    "प्रकाश संश्लेषण",
    "diabetes",
    "मधुमेह",
    "blood pressure",
    "रक्तचाप",
    "capital of france",
    "france"
]

results = {q: [] for q in queries_to_check}

with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        p = json.loads(line)
        text = p.get('text', '')
        q_text = p.get('query', '')
        combined = (text + " " + q_text).lower()
        for q in queries_to_check:
            if q.lower() in combined:
                results[q].append({
                    'id': p.get('id'),
                    'lang': p.get('lang'),
                    'query': p.get('query'),
                    'text': text[:120]
                })

for q, hits in results.items():
    print(f"\n==================================================")
    print(f"Search for: '{q}' (Found: {len(hits)} passages in corpus)")
    print(f"==================================================")
    for h in hits[:5]:
        print(f"  [{h['lang']}] Query: '{h['query']}'")
        print(f"       Text:  {h['text']}...")
