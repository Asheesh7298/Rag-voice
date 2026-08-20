import urllib.request
import urllib.parse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

ENDPOINT = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

test_queries = [
    "भारत की राजधानी क्या है?",
    "भारताची राजधानी कोणती आहे?",
    "what is photosynthesis?",
    "what are symptoms of diabetes?",
    "what is a normal blood pressure reading?",
    "what is the capital of France?"
]

for q in test_queries:
    data = urllib.parse.urlencode({'query': q}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        res = {'error': str(e)}

    print(f"\n==================================================")
    print(f"Query: {q}")
    print(f"Answer: {res.get('answer')}")
    print(f"Confidence: {res.get('confidence')}")
    print(f"Guardrail: {res.get('guardrail_triggered')}")
    print(f"Timings: {res.get('timings_ms')}")
    srcs = res.get("sources", [])
    if srcs:
        print(f"Sources top 1: {srcs[0].get('text', '')[:100]}...")
    else:
        print("Sources: []")
