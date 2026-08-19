"""
Debug script: for queries that get declined with low_qa_confidence,
show the ACTUAL score that was computed (even though it's below threshold),
so we can see the real score distribution and pick a sane threshold.

This requires modal_app.py to log scores even on decline -- see prompt below.
Run: python scripts/debug_declines.py
"""
import json, urllib.request, urllib.parse

MODAL_URL = "https://ac161050--voice-rag-voicerag-fastapi-app.modal.run"

TEST_QUERIES = [
    "what is photosynthesis?",
    "what are symptoms of diabetes?",
    "पानी का क्वथनांक कितना होता है?",
    "सौर मंडल में कितने ग्रह हैं?",
    "महाराष्ट्राची राजधानी कोणती आहे?",
    "सूर्याभोवती किती ग्रह फिरतात?",
    "who won the 2024 presidential election?",
    "what is the capital of France?",  # known good, for comparison
]

def post_query(query):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        f"{MODAL_URL}/query", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

for q in TEST_QUERIES:
    r = post_query(q)
    guardrail = r.get("guardrail_triggered")
    conf = r.get("confidence", "N/A")
    answer = r.get("answer", "")[:80]
    print(f"\nQ: {q}")
    print(f"  guardrail: {guardrail}")
    print(f"  confidence: {conf}")
    print(f"  answer: {answer}")