import json

passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

print(f"Total passages: {len(passages)}")

# Count passages with non-empty answers and queries
valid_qa = []
for p in passages:
    q = p.get('query', '').strip()
    ans = p.get('answers', [])
    text = p.get('text', '').strip()
    lang = p.get('lang', '')
    if q and ans and len(ans) > 0 and ans[0].strip() and ans[0].strip() != "No Answer Present." and len(text) > 50:
        valid_qa.append({
            'query_id': p.get('query_id'),
            'lang': lang,
            'query': q,
            'gold_answer': ans[0].strip(),
            'passage_snippet': text[:120]
        })

print(f"Total valid QA pairs with real answers: {len(valid_qa)}")

# Group by language
by_lang = {}
for item in valid_qa:
    l = item['lang']
    by_lang.setdefault(l, []).append(item)

for l, items in by_lang.items():
    print(f"Language '{l}': {len(items)} questions with gold answers")

# Sample 40 diverse questions across Hindi, Marathi, English, Telugu, etc.
sample_40 = []
for l in ['hi', 'mr', 'en', 'te', 'ta', 'bn', 'gu']:
    if l in by_lang:
        sample_40.extend(by_lang[l][:8])

print(f"Sampled {len(sample_40)} questions across languages.")
with open('data/sample_40_eval.json', 'w', encoding='utf-8') as f:
    json.dump(sample_40[:40], f, ensure_ascii=False, indent=2)
print("Saved to data/sample_40_eval.json")
