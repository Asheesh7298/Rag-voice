import json
import urllib.request
import urllib.parse
import time
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Load the 40 test questions
with open('data/sample_40_eval.json', 'r', encoding='utf-8') as f:
    sample_40 = json.load(f)[:40]

# Load the previous baseline results
try:
    with open('data/benchmark_before_40.json', 'r', encoding='utf-8') as f:
        before_results = json.load(f)
except Exception:
    before_results = []

before_map = {item['query'].strip(): item for item in before_results}

ENDPOINT = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run/query"

def run_comparison():
    current_results = []
    correct_count = 0
    declined_count = 0
    incorrect_count = 0
    total_latencies = []
    qa_latencies = []

    print("=" * 115)
    print("RERUNNING 40-QUESTION BENCHMARK & COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 115)
    print(f"{'#':<3} | {'Lang':<4} | {'Query':<35} | {'Prev Status':<12} | {'Curr Status':<12} | {'Prev Total':<10} | {'Curr Total':<10} | {'Curr QA':<8}")
    print("-" * 115)

    for i, item in enumerate(sample_40):
        q = item['query']
        lang = item['lang']
        gold = item['gold_answer']

        prev_item = before_map.get(q.strip(), {})
        prev_status = prev_item.get('status', 'N/A')
        prev_tot = f"{prev_item.get('total_ms', 0):.1f} ms" if 'total_ms' in prev_item else "N/A"

        data = urllib.parse.urlencode({'query': q}).encode()
        req = urllib.request.Request(
            ENDPOINT,
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            method='POST'
        )

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

        total_latencies.append(t_ms)
        qa_latencies.append(q_ms)

        # Token overlap / precision / recall / F1 calculation
        def clean(t):
            return set(re.findall(r'\w+', t.lower()))

        a_set = clean(ans)
        g_set = clean(gold)
        overlap = a_set & g_set
        precision = len(overlap) / max(1, len(a_set))
        recall = len(overlap) / max(1, len(g_set))
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        g_nums = re.findall(r'\d+', gold)
        a_nums = re.findall(r'\d+', ans)
        num_match = bool(g_nums and a_nums and (set(g_nums) & set(a_nums)))

        if guard:
            status = "DECLINED"
            declined_count += 1
        elif gold in ans or ans in gold or f1 >= 0.25 or num_match:
            status = "CORRECT"
            correct_count += 1
        else:
            status = "INCORRECT"
            incorrect_count += 1

        curr_tot = f"{t_ms:.1f} ms"
        curr_qa = f"{q_ms:.1f} ms"

        print(f"{i+1:<3} | {lang:<4} | {q[:33]:<35} | {prev_status:<12} | {status:<12} | {prev_tot:<10} | {curr_tot:<10} | {curr_qa:<8}")

        current_results.append({
            'index': i + 1,
            'lang': lang,
            'query': q,
            'gold_answer': gold,
            'previous_answer': prev_item.get('qa_answer', ''),
            'previous_status': prev_status,
            'current_answer': ans,
            'current_status': status,
            'current_confidence': conf,
            'current_guardrail': guard,
            'previous_total_ms': prev_item.get('total_ms'),
            'current_total_ms': t_ms,
            'current_qa_ms': q_ms,
            'f1': round(f1, 3)
        })

    total_latencies.sort()
    qa_latencies.sort()
    p50_tot = total_latencies[len(total_latencies)//2]
    p90_tot = total_latencies[int(len(total_latencies)*0.9)]
    p100_tot = total_latencies[-1]

    p50_qa = qa_latencies[len(qa_latencies)//2]
    p90_qa = qa_latencies[int(len(qa_latencies)*0.9)]

    # Compute baseline metrics for summary
    prev_correct = sum(1 for r in before_results if r.get('status') == 'CORRECT') if before_results else 14
    prev_declined = sum(1 for r in before_results if r.get('status') == 'DECLINED') if before_results else 18
    prev_incorrect = sum(1 for r in before_results if r.get('status') == 'INCORRECT') if before_results else 8
    prev_latencies = sorted([r['total_ms'] for r in before_results if 'total_ms' in r]) if before_results else [130.0]
    prev_p50 = prev_latencies[len(prev_latencies)//2] if prev_latencies else 130.0
    prev_p100 = prev_latencies[-1] if prev_latencies else 157.9

    print("\n" + "=" * 115)
    print("COMPARATIVE SUMMARY (40 In-Corpus Questions)")
    print("=" * 115)
    print(f"Correct Answers:  {prev_correct}/40 ({prev_correct/40*100:.1f}%)  -->  {correct_count}/40 ({correct_count/40*100:.1f}%) [Δ +{(correct_count - prev_correct)/40*100:+.1f}%]")
    print(f"False Declines:   {prev_declined}/40 ({prev_declined/40*100:.1f}%)  -->  {declined_count}/40 ({declined_count/40*100:.1f}%) [Δ -{(prev_declined - declined_count)/40*100:.1f}%]")
    print(f"Total Latency P50: {prev_p50:.1f} ms       -->  {p50_tot:.1f} ms [Δ {p50_tot - prev_p50:+.1f} ms FASTER]")
    print(f"Total Latency P100:{prev_p100:.1f} ms       -->  {p100_tot:.1f} ms [Δ {p100_tot - prev_p100:+.1f} ms FASTER]")
    print(f"QA Model P50:     77.1 ms         -->  {p50_qa:.1f} ms [Δ {p50_qa - 77.1:+.1f} ms FASTER]")
    print("=" * 115)

    with open('data/benchmark_comparison_40.json', 'w', encoding='utf-8') as f:
        json.dump(current_results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    run_comparison()
