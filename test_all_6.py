import urllib.request
import urllib.parse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

ENDPOINT = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run/query"

test_queries = [
    "भारत की राजधानी क्या है?",
    "भारताची राजधानी कोणती आहे?",
    "what is photosynthesis?",
    "what are symptoms of diabetes?",
    "what is a normal blood pressure reading?",
    "what is the capital of France?"
]

print("=" * 90)
print("TESTING USER'S 6 SPECIFIC QUERIES")
print("=" * 90)

for q in test_queries:
    data = urllib.parse.urlencode({'query': q}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        res = {'error': str(e)}

    print(f"\nQuery:     {q}")
    print(f"Answer:    {res.get('answer')}")
    print(f"Conf:      {res.get('confidence')}")
    print(f"Latency:   {res.get('timings_ms', {}).get('total_ms')} ms")
print("=" * 90)
