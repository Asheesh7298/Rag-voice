import urllib.request
import urllib.parse
import json

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"
test_queries = [
    ("HI", "सबसे बड़ा उड़ने वाला सरीसृप अब तक"),
    ("MR", "फ्रान्सचे सध्याचे चलन काय आहे"),
    ("EN", "nyu tuition cost")
]

print("=" * 60)
print(f"TESTING NEW DEPLOYMENT: {MODAL_URL}")
print("=" * 60)

for lang, q in test_queries:
    data = urllib.parse.urlencode({"query": q}).encode("utf-8")
    req = urllib.request.Request(
        f"{MODAL_URL}/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        ans = res.get("answer", "")[:50].replace("\n", " ")
        timings = res.get("timings_ms", {})
        tot = timings.get("total_ms", 0)
        sch = timings.get("search_ms", 0)
        qa = timings.get("qa_ms", 0)
        print(f"[{lang}] Q: {q}")
        print(f"     Ans: {ans}")
        print(f"     Latency: {tot:.1f}ms (Search: {sch:.1f}ms, QA: {qa:.1f}ms)")
        print("-" * 60)
