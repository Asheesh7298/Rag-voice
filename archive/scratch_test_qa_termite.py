import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Let's find the retrieved chunk for "दीमक बॉन्ड की कीमत कितनी होती है?" in passages.jsonl
passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

q = "दीमक बॉन्ड की कीमत कितनी होती है?"
matching = [p for p in passages if "दीमक" in p.get('text', '') or "कीमत" in p.get('text', '')]
print(f"Found {len(matching)} matching passages in dataset.")
for p in matching[:3]:
    print("---")
    print(f"Query in passage: {p.get('query')}")
    print(f"Text: {p.get('text')[:180]}")
    print(f"Answers: {p.get('answers')}")

# Now let's test QA model extraction on this text
model_name = "deepset/xlm-roberta-base-squad2"
tok = AutoTokenizer.from_pretrained(model_name)
qa = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa.eval()

if matching:
    context = matching[0]['text']
    inputs = tok(q, context, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        out = qa(**inputs)
    s_idx = int(torch.argmax(out.start_logits[0]))
    e_idx = int(torch.argmax(out.end_logits[0]))
    if e_idx < s_idx:
        e_idx = s_idx
    ans = tok.decode(inputs['input_ids'][0][s_idx:e_idx+1], skip_special_tokens=True)
    s_prob = float(F.softmax(out.start_logits[0], dim=-1)[s_idx])
    e_prob = float(F.softmax(out.end_logits[0], dim=-1)[e_idx])
    score = s_prob * e_prob
    print("\n--- QA Model Result ---")
    print(f"start_idx: {s_idx}, end_idx: {e_idx}")
    print(f"start_prob: {s_prob:.6f}, end_prob: {e_prob:.6f}, score: {score:.6f}")
    print(f"Extracted answer: {ans!r}")
