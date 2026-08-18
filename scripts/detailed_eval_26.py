import json
import re
import sys
import urllib.request
import urllib.parse
import time

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load gold answers and queries from passages.jsonl
passages = []
with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

# Map normalized query -> passage info
gold_map = {}
for p in passages:
    q_norm = p.get('query', '').strip().lower()
    if q_norm:
        gold_map[q_norm] = {
            'query_id': p.get('query_id'),
            'lang': p.get('lang'),
            'text': p.get('text'),
            'answers': p.get('answers', []),
            'is_selected': p.get('is_selected', False),
        }

TEST_QUERIES = [
    # Hindi (10)
    ("hi", "प्रकाश संश्लेषण क्या है?"),
    ("hi", "डीएनए की संरचना क्या है?"),
    ("hi", "परमाणु के मुख्य भाग कौन से हैं?"),
    ("hi", "मधुमेह के लक्षण क्या हैं?"),
    ("hi", "रक्तचाप सामान्य कितना होना चाहिए?"),
    ("hi", "हाइपोथायरायडिज्म का क्या अर्थ है?"),
    ("hi", "प्रति वर्ग फुट टाइल स्थापना की लागत क्या है?"),
    ("hi", "इलिनोइस में एक एलपीएन प्रति घंटे कितना कमाता है?"),
    ("hi", "भारत की राजधानी क्या है?"),
    ("hi", "अमेरिका की स्वतंत्रता कब हुई?"),

    # Marathi (7)
    ("mr", "प्रकाशसंश्लेषण म्हणजे काय?"),
    ("mr", "डीएनएची रचना काय आहे?"),
    ("mr", "मधुमेहाची लक्षणे कोणती आहेत?"),
    ("mr", "हायपोथायरॉईडीझम म्हणजे काय?"),
    ("mr", "टाइल बसवण्याचा खर्च किती आहे?"),
    ("mr", "इलिनॉयमध्ये एलपीएन दर तासाला किती कमावतो?"),
    ("mr", "भारताची राजधानी कोणती आहे?"),

    # English (9)
    ("en", "what is photosynthesis?"),
    ("en", "what are the main parts of an atom?"),
    ("en", "what are symptoms of diabetes?"),
    ("en", "what does hypothyroidism mean?"),
    ("en", "what is a normal blood pressure reading?"),
    ("en", "how much does an LPN earn per hour in Illinois?"),
    ("en", "what is the cost of tile installation per square foot?"),
    ("en", "what is the capital of France?"),
    ("en", "when did America gain independence?"),
]

ENDPOINT = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run/query"

def evaluate():
    results = []
    for lang, q in TEST_QUERIES:
        q_norm = q.strip().lower()
        
        # Check if gold exists in dataset
        gold_info = gold_map.get(q_norm)
        if not gold_info:
            # Fuzzy match query
            for k, v in gold_map.items():
                if q_norm.replace('?', '').strip() in k or k in q_norm:
                    gold_info = v
                    break

        gold_ans = gold_info['answers'][0] if gold_info and gold_info.get('answers') else None
        gold_passage = gold_info['text'] if gold_info else None
        in_dataset = gold_info is not None

        # Call live API
        data = urllib.parse.urlencode({'query': q}).encode()
        req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')
        
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            payload = {'answer': f'ERROR: {e}', 'confidence': 0.0, 'guardrail_triggered': 'error', 'sources': [], 'timings_ms': {}}
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        ans = payload.get('answer', '')
        conf = payload.get('confidence', 0.0)
        guard = payload.get('guardrail_triggered')
        sources = payload.get('sources', [])
        timings = payload.get('timings_ms', {})
        total_ms = timings.get('total_ms', wall_ms)
        qa_ms = timings.get('qa_ms', 0.0)

        # Check retrieval: is gold passage in retrieved sources?
        retrieved_status = "N/A (Out of dataset)"
        if in_dataset and gold_passage:
            retrieved = False
            for s in sources:
                if s.get('text', '')[:60] in gold_passage or gold_passage[:60] in s.get('text', ''):
                    retrieved = True
                    break
            retrieved_status = "YES" if retrieved else "NO"

        # Classify answer correctness
        # If declined
        if guard:
            classification = "DECLINED"
            cause = "N/A"
            if not in_dataset:
                cause = "Expected decline (out of dataset)"
            else:
                cause = "D: Correct answer/passage existed but guardrail rejected"
        else:
            # Check if answer matches gold semantically or token-wise
            def normalize(t):
                return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()
            
            p_toks = set(normalize(ans).split())
            g_toks = set(normalize(gold_ans).split()) if gold_ans else set()
            
            common = p_toks & g_toks
            prec = len(common) / max(1, len(p_toks))
            rec = len(common) / max(1, len(g_toks))
            f1 = 2 * prec * rec / max(1e-6, prec + rec)

            # Domain specific semantic check
            is_sem_correct = False
            if gold_ans:
                if f1 >= 0.35 or gold_ans in ans or ans in gold_ans:
                    is_sem_correct = True
                # Numbers check (e.g. costs, dates)
                g_nums = re.findall(r'\d+', gold_ans)
                a_nums = re.findall(r'\d+', ans)
                if g_nums and a_nums and set(g_nums) == set(a_nums):
                    is_sem_correct = True
            
            if is_sem_correct:
                classification = "CORRECT"
                cause = "None"
            else:
                classification = "INCORRECT"
                if not in_dataset:
                    cause = "Dataset does not contain gold answer (Out of scope)"
                elif retrieved_status == "NO":
                    cause = "A: Gold passage not retrieved"
                else:
                    cause = "C: Gold passage reached QA but QA selected wrong span"

        results.append({
            'lang': lang,
            'query': q,
            'in_dataset': in_dataset,
            'gold_answer': gold_ans or "[Out of dataset]",
            'retrieved': retrieved_status,
            'qa_answer': ans[:80],
            'classification': classification,
            'confidence': conf,
            'guardrail': guard,
            'total_ms': total_ms,
            'qa_ms': qa_ms,
            'cause': cause
        })

    return results

if __name__ == '__main__':
    res = evaluate()
    print(json.dumps(res, ensure_ascii=False, indent=2))
