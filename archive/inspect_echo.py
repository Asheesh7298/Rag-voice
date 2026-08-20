import urllib.request
import re

req = urllib.request.Request(
    'https://echo.omchillure.space/assets/index-B3OMOu1w.js',
    headers={'User-Agent': 'Mozilla/5.0'}
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        js = resp.read().decode('utf-8')
        
    print("JS bundle size:", len(js), "bytes")
    
    # 1. URLs & Endpoints
    urls = set(re.findall(r'https?://[^\s"\'`<>]+', js))
    print("\n--- Extracted Endpoints / URLs ---")
    for u in sorted(urls):
        print("  -", u)
        
    # 2. Keywords and UI Text
    terms = ['latency', 'ms', 'rag', 'sarvam', 'gpu', 'vector', 'guardrail', 'indic', 'hindi', 'marathi', 'p50', 'modal', 'faiss', 'fastapi', 'whisper', 'groq', 'together', 'gemini']
    print("\n--- Key Concepts & Snippets ---")
    for t in terms:
        matches = [m.start() for m in re.finditer(r'\b' + t + r'\b', js, re.IGNORECASE)]
        print(f"Keyword '{t}': {len(matches)} matches")
        for idx in matches[:2]:
            snippet = js[max(0, idx - 40):min(len(js), idx + 80)].replace('\n', ' ')
            print(f"    snippet: ...{snippet}...")

except Exception as e:
    print("Error:", e)
