import json
import random

# Load passages with queries
passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        p = json.loads(line)
        if p.get('query') and len(p['query'].strip()) > 5:
            passages.append(p)

print(f"Total passages with query: {len(passages)}")

# Group by language
lang_groups = {}
for p in passages:
    lang = p.get('lang', 'unknown')
    if lang not in lang_groups:
        lang_groups[lang] = []
    lang_groups[lang].append(p)

print("Passage counts by language:")
for lang, items in lang_groups.items():
    print(f"  {lang}: {len(items)}")

# Select a balanced set of 50 questions across languages
random.seed(42)
selected_50 = []
langs = sorted(list(lang_groups.keys()))
per_lang = max(1, 50 // len(langs))

for lang in langs:
    items = lang_groups[lang]
    sample_size = min(len(items), 4)
    sampled = random.sample(items, sample_size)
    for item in sampled:
        selected_50.append({
            'query': item['query'].strip(),
            'lang': item['lang'],
            'passage_id': item.get('id'),
            'passage_text': item.get('text', ''),
            'gold_answer': item.get('answer', '') or item.get('gold_answer', '')
        })

# Fill remaining to make exactly 50
remaining = [p for p in passages if p['query'].strip() not in {s['query'] for s in selected_50}]
random.shuffle(remaining)
while len(selected_50) < 50 and remaining:
    item = remaining.pop(0)
    selected_50.append({
        'query': item['query'].strip(),
        'lang': item['lang'],
        'passage_id': item.get('id'),
        'passage_text': item.get('text', ''),
        'gold_answer': item.get('answer', '') or item.get('gold_answer', '')
    })

selected_50 = selected_50[:50]
print(f"Selected {len(selected_50)} evaluation questions.")

with open('data/sample_50_eval.json', 'w', encoding='utf-8') as f:
    json.dump(selected_50, f, ensure_ascii=False, indent=2)

print("Saved to data/sample_50_eval.json")
