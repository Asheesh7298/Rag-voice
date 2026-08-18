import json
import urllib.request
import urllib.parse
import time
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('data/sample_40_eval.json', 'r', encoding='utf-8') as f:
    sample_40 = json.load(f)[:40]

ENDPOINT = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run/query"

def run_eval():
    results = []
    correct_count = 0
    declined_count = 0
    incorrect_count = 0
    total_latency = []
    qa_latency = []

    print(f"Evaluating {len(sample_40)} in-corpus questions on live endpoint...\n")
    print(f"{'#':<3} | {'Lang':<4} | {'Query':<40} | {'Status':<10} | {'Conf':<6} | {'Total ms':<8} | {'QA ms':<6}")
    print("-" * 90)

    for i, item in enumerate(sample_40):
        q = item['query']
        lang = item['lang']
        gold = item['gold_answer']
        snippet = item['passage_snippet']

        data = urllib.parse.urlencode({'query': q}).encode()
        req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            res = {'answer': f"ERROR: {e}", 'confidence': 0.0, 'guardrail_triggered': 'error', 'timings_ms': {}}
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        ans = res.get('answer', '')
        conf = res.get('confidence', 0.0)
        guard = res.get('guardrail_triggered')
        timings = res.get('timings_ms', {})
        t_ms = timings.get('total_ms', wall_ms)
        q_ms = timings.get('qa_ms', 0.0)

        total_latency.append(t_ms)
        qa_latency.append(q_ms)

        # Check semantic / token overlap with gold answer
        def clean(t):
            return set(re.findall(r'\w+', t.lower()))
        
        a_set = clean(ans)
        g_set = clean(gold)
        overlap = a_set & g_set
        precision = len(overlap) / max(1, len(a_set))
        recall = len(overlap) / max(1, len(g_set))
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        # Check if numbers or key entities match
        g_nums = re.findall(r'\d+', gold)
        a_nums = re.findall(r'\d+', ans)
        num_match = bool(g_nums and a_nums and (set(g_nums) & set(a_nums)))

        is_correct = False
        if guard:
            status = "DECLINED"
            declined_count += 1
        elif gold in ans or ans in gold or f1 >= 0.3 or num_match:
            status = "CORRECT"
            is_correct = True
            correct_count += 1
        else:
            status = "INCORRECT"
            incorrect_count += 1

        print(f"{i+1:<3} | {lang:<4} | {q[:38]:<40} | {status:<10} | {conf:<6.2f} | {t_ms:<8.1f} | {q_ms:<6.1f}")
        
        results.append({
            'index': i + 1,
            'lang': lang,
            'query': q,
            'gold_answer': gold,
            'qa_answer': ans,
            'status': status,
            'confidence': conf,
            'guardrail': guard,
            'total_ms': t_ms,
            'qa_ms': q_ms,
            'f1': round(f1, 3)
        })

    total_latency.sort()
    qa_latency.sort()
    p50_tot = total_latency[len(total_latency)//2]
    p90_tot = total_latency[int(len(total_latency)*0.9)]
    p100_tot = total_latency[-1]

    p50_qa = qa_latency[len(qa_latency)//2]
    p90_qa = qa_latency[int(len(qa_latency)*0.9)]

    print("\n" + "=" * 90)
    print(f"BENCHMARK SUMMARY (40 In-Corpus Questions):")
    print(f"Correct: {correct_count}/40 ({correct_count/40*100:.1f}%)")
    print(f"Incorrect: {incorrect_count}/40 ({incorrect_count/40*100:.1f}%)")
    print(f"Declined: {declined_count}/40 ({declined_count/40*100:.1f}%)")
    print(f"Total Latency: P50 = {p50_tot:.1f} ms | P90 = {p90_tot:.1f} ms | P100 = {p100_tot:.1f} ms")
    print(f"QA Latency:    P50 = {p50_qa:.1f} ms | P90 = {p90_qa:.1f} ms")
    print("=" * 90)

    with open('data/benchmark_before_40.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    run_eval()
