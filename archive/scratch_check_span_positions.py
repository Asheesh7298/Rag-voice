import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/sample_40_eval.json', 'r', encoding='utf-8') as f:
    sample_40 = json.load(f)

passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

q_to_passage = {p['query'].strip().lower(): p for p in passages if p.get('query')}

print(f"Total test questions: {len(sample_40)}")
for i, item in enumerate(sample_40):
    q = item['query'].strip().lower()
    gold = item['gold_answer'].strip()
    p_data = q_to_passage.get(q)
    if p_data:
        p_text = p_data.get('text', '')
        pos = p_text.find(gold)
        char_len = len(p_text)
        word_count = len(p_text.split())
        words_before = len(p_text[:pos].split()) if pos != -1 else -1
        print(f"Q{i+1:<2} [{item['lang']}]: Gold present: {pos != -1!s:<5} | Words before gold: {words_before:<3} | Total passage words: {word_count}")
