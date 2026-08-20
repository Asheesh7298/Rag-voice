import urllib.request, json

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode('utf-8')
            return json.loads(data)
    except Exception as e:
        return {"error": str(e)}

def post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
            return json.loads(data)
    except Exception as e:
        return {"error": str(e)}

print("=== 1. Health Endpoint ===")
print(json.dumps(get('https://echo.omchillure.space/api/health'), indent=2))

print("\n=== 2. Metrics Endpoint ===")
print(json.dumps(get('https://echo.omchillure.space/api/metrics'), indent=2))

print("\n=== 3. Query /api/ask ===")
print(json.dumps(post('https://echo.omchillure.space/api/ask', {"query": "what county is columbus city in"}), indent=2))
