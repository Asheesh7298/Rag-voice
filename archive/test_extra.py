import urllib.request
import urllib.parse
import json

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

extra = [
    ("HI", "साइबोर्ग डीसी फिल्म कब आ रही है"),
    ("HI", "दिल्ली कैलिफोर्निया कौन्टी है"),
    ("EN", "how much does it cost to change a jeep alternator"),
    ("MR", "अल्फा हेलिक्स कुठे आढळते ज्या प्रथिन संघटनेच्या स्तरावर आहे")
]

for lang, q in extra:
    data = urllib.parse.urlencode({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"[{lang}] {q} -> {res.get('answer')}")
