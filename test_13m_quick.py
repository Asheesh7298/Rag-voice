import urllib.request
import urllib.parse
import json
import time

MODAL_BASE = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

print("1. Testing /health...")
req = urllib.request.Request(f"{MODAL_BASE}/health")
with urllib.request.urlopen(req, timeout=60) as r:
    print(r.read().decode("utf-8"))

print("\n2. Testing /query (English)...")
data = urllib.parse.urlencode({"query": "what county is columbus city in"}).encode("utf-8")
req = urllib.request.Request(f"{MODAL_BASE}/query", data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
with urllib.request.urlopen(req, timeout=60) as r:
    res = json.loads(r.read().decode("utf-8"))
    print("Answer:", res.get("answer"))
    print("Timings:", res.get("timings_ms"))
