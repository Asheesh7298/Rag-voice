import urllib.request
import urllib.parse
import json

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

test_candidates = [
    ("EN", "what county is columbus city in"),
    ("EN", "average cost dental implant"),
    ("HI", "फ्लोरिडा में गोल्फ कार्ट चलाने की उम्र क्या है"),
    ("HI", "कोपर शहर से हॉलीवुड कितना दूर है"),
    ("MR", "पृथ्वी किती जुनी आहे"),
    ("MR", "पार्किंग ब्रेक म्हणजे ई ब्रेक आहे")
]

for lang, q in test_candidates:
    data = urllib.parse.urlencode({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        ans = res.get("answer", "")
        guard = res.get("guardrail_triggered")
        print(f"[{lang}] Question: {q}")
        print(f"     Answer:   {ans}")
        print(f"     Guard:    {guard}")
        print("-" * 65)
