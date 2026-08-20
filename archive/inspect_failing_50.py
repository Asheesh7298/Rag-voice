import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/benchmark_50_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

failing = [r for r in results if r['status'] != 'CORRECT']
print(f"Total failing: {len(failing)}")
for r in failing:
    print(f"\n--- [Q{r['index']}] ({r['lang']}) {r['query']} ---")
    print(f"  Gold Answer:   {r['gold_answer']}")
    print(f"  Model Answer:  {r['model_answer']}")
    print(f"  Passage Text:  {r['passage_text']}")
    print(f"  Sources:       {r['sources']}")
