"""
Multi-language test: 2-3 questions per language, latency breakdown per query.
Run: python scripts/multilang_test.py
"""
import json, urllib.request, urllib.parse, time, statistics

MODAL_URL = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run"

# 2-3 real questions per language from IndicMSMARCO domain
QUERIES = {
    "hi": [
        "एक न्यूक्लियोटाइड का कौन सा एक भाग अलग होता है?",
        "प्रति वर्ग फुट टाइल स्थापना की लागत क्या है?",
        "हाइपोथायरायडिज्म का क्या अर्थ है?",
    ],
    "bn": [
        "ডিএনএ-তে নিউক্লিওটাইডের কোন অংশ আলাদা?",
        "পটাশিয়াম নাইট্রেট কোথায় পাওয়া যায়?",
    ],
    "ta": [
        "டைல் நிறுவலுக்கான சதுர அடி செலவு என்ன?",
        "நியூக்ளியோடைடின் எந்த பகுதி வேறுபட்டது?",
    ],
    "te": [
        "టైల్ ఇన్స్టాలేషన్ ధర ఏమిటి?",
        "హైపోథైరాయిడిజం అంటే ఏమిటి?",
    ],
    "mr": [
        "न्यूक्लियोटाइडचा कोणता भाग वेगळा असतो?",
        "टाइल बसवण्याचा खर्च किती आहे?",
    ],
    "gu": [
        "ન્યુક્લિઓટાઇડનો કયો ભાગ અલગ છે?",
        "ટાઇલ ઇન્સ્ટોલેશનની કિંમત શું છે?",
    ],
    "kn": [
        "ನ್ಯೂಕ್ಲಿಯೋಟೈಡ್‌ನ ಯಾವ ಭಾಗ ವಿಭಿನ್ನವಾಗಿದೆ?",
        "ಟೈಲ್ ಅಳವಡಿಕೆಯ ವೆಚ್ಚ ಎಷ್ಟು?",
    ],
    "ml": [
        "ന്യൂക്ലിയോടൈഡിന്റെ ഏത് ഭാഗം വ്യത്യസ്തമാണ്?",
        "ടൈൽ ഇൻസ്റ്റലേഷൻ ചെലവ് എത്രയാണ്?",
    ],
    "pa": [
        "ਨਿਊਕਲੀਓਟਾਈਡ ਦਾ ਕਿਹੜਾ ਹਿੱਸਾ ਵੱਖਰਾ ਹੈ?",
        "ਟਾਈਲ ਲਗਾਉਣ ਦੀ ਕੀਮਤ ਕੀ ਹੈ?",
    ],
    "ne": [
        "न्युक्लिओटाइडको कुन भाग फरक हुन्छ?",
        "टाइल जडानको लागत कति हो?",
    ],
    "or": [
        "ନ୍ୟୁକ୍ଲିଓଟାଇଡ୍ ର କେଉଁ ଭାଗ ଅଲଗା?",
        "ଟାଇଲ୍ ଲଗାଇବା ଖର୍ଚ୍ଚ କେତେ?",
    ],
    "as": [
        "নিউক্লিঅ'টাইডৰ কোন অংশ পৃথক?",
        "টাইল স্থাপনৰ খৰচ কিমান?",
    ],
    "ur": [
        "نیوکلیوٹائیڈ کا کون سا حصہ مختلف ہے؟",
        "ٹائل انسٹالیشن کی لاگت کیا ہے؟",
    ],
}

def post_query(query):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        f"{MODAL_URL}/query", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    result["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result

LANG_NAMES = {
    "hi":"Hindi","bn":"Bengali","ta":"Tamil","te":"Telugu","mr":"Marathi",
    "gu":"Gujarati","kn":"Kannada","ml":"Malayalam","pa":"Punjabi",
    "ne":"Nepali","or":"Odia","as":"Assamese","ur":"Urdu"
}

print(f"Testing {MODAL_URL}")
print("=" * 80)

all_totals = []
all_qa = []
results_by_lang = {}

for lang, queries in QUERIES.items():
    lang_name = LANG_NAMES.get(lang, lang)
    print(f"\n{'='*80}")
    print(f"  {lang_name} ({lang})")
    print(f"{'='*80}")
    lang_results = []

    for q in queries:
        try:
            r = post_query(q)
            t = r.get("timings_ms", {})
            status = f"GUARDRAIL:{r['guardrail_triggered']}" if r.get("guardrail_triggered") else "✅ ANSWERED"
            answer = r.get("answer", "")[:60]
            conf = r.get("confidence", 0)

            print(f"\n  Q: {q[:60]}")
            print(f"  Status: {status}")
            print(f"  Answer: {answer}")
            if not r.get("guardrail_triggered"):
                print(f"  Confidence: {conf:.3f}")
            print(f"  Latency: embed={t.get('embed_ms','?')}ms | search={t.get('search_ms','?')}ms | rerank={t.get('rerank_ms','?')}ms | qa={t.get('qa_ms','?')}ms | total={t.get('total_ms','?')}ms | wall={r['wall_ms']}ms")

            total = t.get("total_ms", 0)
            qa = t.get("qa_ms", 0)
            if total: all_totals.append(total)
            if qa: all_qa.append(qa)
            lang_results.append({"query": q, "status": status, "answer": answer, "total_ms": total})
        except Exception as e:
            print(f"\n  Q: {q[:60]}")
            print(f"  ERROR: {e}")

    results_by_lang[lang] = lang_results

# Summary
def pct(vals, p):
    if not vals: return 0
    s = sorted(vals)
    k = (len(s)-1)*p/100
    f, c = int(k), min(int(k)+1, len(s)-1)
    return round(s[f]+(s[c]-s[f])*(k-f), 1)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
total_q = sum(len(v) for v in QUERIES.values())
answered = sum(1 for lang in results_by_lang.values() for r in lang for k,v in [("status","")] if "ANSWERED" in r.get("status",""))
print(f"Total queries: {total_q} | Answered: {answered} | Declined: {total_q - answered}")
if all_totals:
    print(f"\nLatency (pipeline total_ms) across all languages:")
    print(f"  P50={pct(all_totals,50)}ms  P70={pct(all_totals,70)}ms  P90={pct(all_totals,90)}ms  P100={pct(all_totals,100)}ms")
if all_qa:
    print(f"QA model (qa_ms):")
    print(f"  P50={pct(all_qa,50)}ms  P70={pct(all_qa,70)}ms  P90={pct(all_qa,90)}ms  P100={pct(all_qa,100)}ms")