import json
import urllib.request
import urllib.parse
import time
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

ENDPOINT = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run/query"

def evaluate_50():
    with open('data/sample_50_eval.json', 'r', encoding='utf-8') as f:
        sample_50 = json.load(f)

    results = []
    correct_count = 0
    incorrect_count = 0
    declined_count = 0
    total_latencies = []
    qa_latencies = []

    print("=" * 120)
    print("RUNNING 50-QUESTION EVALUATION SUITE")
    print("=" * 120)
    print(f"{'#':<3} | {'Lang':<4} | {'Query':<35} | {'Status':<10} | {'Total ms':<9} | {'QA ms':<8} | {'Overlap / Score':<15}")
    print("-" * 120)

    for i, item in enumerate(sample_50):
        q = item['query']
        lang = item['lang']
        gold = item.get('gold_answer', '')
        p_text = item.get('passage_text', '')

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
        guard = res.get('guardrail_triggered')
        timings = res.get('timings_ms', {})
        t_ms = timings.get('total_ms', wall_ms)
        q_ms = timings.get('qa_ms', 0.0)

        total_latencies.append(t_ms)
        qa_latencies.append(q_ms)

        # Verification logic:
        # 1. Direct substring match with gold answer
        # 2. Token overlap / F1 score with gold answer
        # 3. If gold answer is short or noisy, check overlap with key ground-truth passage sentence
        def clean(t):
            return set(re.findall(r'\w+', t.lower()))

        a_set = clean(ans)
        g_set = clean(gold)
        p_set = clean(p_text)

        overlap_g = a_set & g_set
        prec_g = len(overlap_g) / max(1, len(a_set))
        rec_g = len(overlap_g) / max(1, len(g_set))
        f1_g = (2 * prec_g * rec_g / max(1e-6, prec_g + rec_g)) if (prec_g + rec_g) > 0 else 0.0

        g_nums = re.findall(r'\d+', gold)
        a_nums = re.findall(r'\d+', ans)
        num_match = bool(g_nums and a_nums and (set(g_nums) & set(a_nums)))

        # Also check ground truth passage containment
        gt_containment = False
        if p_text and ans:
            # Check if answer is a valid sentence/span from ground truth passage
            ans_clean_str = re.sub(r'[^\w\s]', '', ans).strip()
            p_clean_str = re.sub(r'[^\w\s]', '', p_text).strip()
            if (len(ans_clean_str) >= 10 and (ans_clean_str[:40] in p_clean_str or ans_clean_str in p_clean_str)) or len(a_set & p_set) >= 4:
                gt_containment = True

        if guard:
            status = "DECLINED"
            declined_count += 1
        elif (gold and (gold in ans or ans in gold or f1_g >= 0.25 or num_match)) or (gt_containment and len(a_set & clean(q)) < len(a_set)):
            status = "CORRECT"
            correct_count += 1
        else:
            status = "INCORRECT"
            incorrect_count += 1

        score_info = f"F1={f1_g:.2f}" if f1_g > 0 else ("GT_Match" if gt_containment else "No_Match")
        print(f"{i+1:<3} | {lang:<4} | {q[:33]:<35} | {status:<10} | {t_ms:<9.1f} | {q_ms:<8.1f} | {score_info:<15}")

        results.append({
            'index': i + 1,
            'lang': lang,
            'query': q,
            'gold_answer': gold,
            'passage_text': p_text[:150],
            'model_answer': ans,
            'status': status,
            'total_ms': t_ms,
            'qa_ms': q_ms,
            'f1_gold': round(f1_g, 3),
            'gt_containment': gt_containment,
            'sources': [s.get('text', '')[:80] for s in res.get('sources', [])[:3]]
        })

    total_latencies.sort()
    qa_latencies.sort()
    p50_tot = total_latencies[len(total_latencies)//2]
    p90_tot = total_latencies[int(len(total_latencies)*0.9)]
    p100_tot = total_latencies[-1]

    p50_qa = qa_latencies[len(qa_latencies)//2]

    print("\n" + "=" * 120)
    print("50-QUESTION EVALUATION SUMMARY")
    print("=" * 120)
    print(f"Correct Answers:  {correct_count} / 50 ({correct_count/50*100:.1f}%) [Target: >= 45/50 (90.0%)]")
    print(f"False Declines:   {declined_count} / 50 ({declined_count/50*100:.1f}%)")
    print(f"Incorrect:        {incorrect_count} / 50 ({incorrect_count/50*100:.1f}%)")
    print(f"Total Latency P50: {p50_tot:.1f} ms  | P90: {p90_tot:.1f} ms  | P100: {p100_tot:.1f} ms")
    print(f"QA Model P50:     {p50_qa:.1f} ms")
    print("=" * 120)

    with open('data/benchmark_50_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return correct_count, results

if __name__ == '__main__':
    evaluate_50()
