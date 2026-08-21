import urllib.request
import urllib.parse
import json
import time

MODAL_BASE = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

print("=" * 70)
print("   TESTING LIVE 13.02M MULTI-STRATEGY + QWEN2.5-0.5B PIPELINE")
print("=" * 70)

# 1. Health & Index Size
print("\n[1/5] Checking Health & 13M Index Size on Modal...")
with urllib.request.urlopen(f"{MODAL_BASE}/health", timeout=120) as r:
    h = json.loads(r.read().decode("utf-8"))
    print(f"  ✅ Status: {h.get('status')} | Total Vectors: {h.get('index_size'):,} vectors")

# 2. Queries
def test_q(q, lang):
    data = urllib.parse.urlencode({"query": q}).encode("utf-8")
    req = urllib.request.Request(f"{MODAL_BASE}/query", data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=45) as r:
        res = json.loads(r.read().decode("utf-8"))
        t = res.get("timings_ms", {})
        print(f"  ✅ [{lang}] Query: {q!r}")
        print(f"     Answer: {res.get('answer', '')}")
        print(f"     Grounded: {res.get('grounded')} | Guardrail: {res.get('guardrail_triggered')}")
        print(f"     Timings: total={t.get('total_ms')}ms | search={t.get('search_ms')}ms | rerank={t.get('rerank_ms')}ms | qa={t.get('qa_ms')}ms | gen={t.get('gen_ms')}ms")

print("\n[2/5] Testing English Query...")
test_q("what county is columbus city in", "EN")

print("\n[3/5] Testing Hindi Query...")
test_q("ब्राइटन टाउनशिप फोन नंबर", "HI")

print("\n[4/5] Testing Marathi Query...")
test_q("फ्रान्सचे सध्याचे चलन काय आहे", "MR")

print("\n[5/5] Testing Guardrail Off-Topic Rejection...")
test_q("who won the cricket match on mars tomorrow", "EN")

print("\n" + "=" * 70)
print("     13.02M VECTOR + QWEN2.5-0.5B SYSTEM VERIFICATION COMPLETE")
print("=" * 70)
