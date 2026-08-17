import json

rows = [json.loads(l) for l in open('data/processed/passages.jsonl', 'r', encoding='utf-8')]

# Find the exact passage for hi-10440
for r in rows:
    if r['query_id'] == 'hi-10440':
        print("=== PASSAGE hi-10440 ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print()

# Also check ne-10440 (Nepali version of same query)
for r in rows:
    if r['query_id'] == 'ne-10440':
        print("=== PASSAGE ne-10440 ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print()

# Show dataset distribution
from collections import Counter
lang_counts = Counter(r['lang'] for r in rows)
print("=== LANGUAGE DISTRIBUTION ===")
for lang, count in sorted(lang_counts.items()):
    print(f"  {lang}: {count}")

# Check is_selected distribution
selected = sum(1 for r in rows if r.get('is_selected'))
print(f"\nis_selected=True: {selected}/{len(rows)} ({100*selected/len(rows):.1f}%)")

# Average text length by language
from statistics import mean
for lang in sorted(lang_counts.keys()):
    lens = [len(r['text']) for r in rows if r['lang'] == lang]
    print(f"  {lang}: avg text len = {mean(lens):.0f} chars")
