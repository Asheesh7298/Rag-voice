import json
import urllib.request
import urllib.parse
import sys

sys.stdout.reconfigure(encoding='utf-8')

q = "दीमक बॉन्ड की कीमत कितनी होती है?"
ENDPOINT = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

data = urllib.parse.urlencode({'query': q}).encode()
req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')

with urllib.request.urlopen(req, timeout=30) as resp:
    res = json.loads(resp.read().decode('utf-8'))

print("Query response:")
print(json.dumps(res, ensure_ascii=False, indent=2))
