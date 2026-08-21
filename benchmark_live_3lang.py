import urllib.request
import json
import time

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

tests = [
    ("what county is columbus city in", "ENGLISH"),
    ("ब्राइटन टाउनशिप फोन नंबर", "HINDI"),
    ("फ्रान्सचे सध्याचे चलन काय आहे", "MARATHI"),
]

print("=" * 75)
print("     TESTING LIVE MODAL A100 ENDPOINT (13.02M VECTORS + QWEN2.5-0.5B)")
print("=" * 75)

for q, label in tests:
    print(f"\n--- [{label} QUERY]: \"{q}\" ---")
    payload = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=payload, headers={"Content-Type": "application/json"})
    
    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            e2e_ms = round((time.perf_counter() - t_start) * 1000, 2)
            
            print(f"  💬 Answer:     {data.get('answer')}")
            print(f"  🛡️ Grounded:   {data.get('grounded')} | Guardrail: {data.get('guardrail_triggered')}")
            t = data.get("timings_ms", {})
            print(f"  ⚡ Server Timings: Total={t.get('total_ms')}ms | Dense_Search={t.get('search_ms')}ms | Qwen_Gen={t.get('gen_ms')}ms | Embed={t.get('embed_ms')}ms")
            print(f"  🌐 Total Client E2E (incl network): {e2e_ms}ms")
            if data.get("sources"):
                print(f"  📖 Top Fact:   {data['sources'][0]['text'][:110]}...")
    except Exception as e:
        print(f"  ❌ Request Error: {e}")

print("\n" + "=" * 75)
print("                       ALL 3 TESTS COMPLETE")
print("=" * 75)
