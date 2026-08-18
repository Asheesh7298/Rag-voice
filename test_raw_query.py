import urllib.request
import urllib.parse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

ENDPOINT = "https://prkhr-g--voice-rag-voicerag-fastapi-app.modal.run/query"
data = urllib.parse.urlencode({'query': 'भारत की राजधानी क्या है?'}).encode()
req = urllib.request.Request(ENDPOINT, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')
        print("RAW RESPONSE:")
        print(raw)
        res = json.loads(raw)
        print("PARSED:", res)
except Exception as e:
    print("ERROR:", e)
