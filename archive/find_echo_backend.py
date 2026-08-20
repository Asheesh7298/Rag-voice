import urllib.request, re

req = urllib.request.Request('https://echo.omchillure.space/assets/index-B3OMOu1w.js', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    js = resp.read().decode('utf-8')

# Search for API route patterns, fetch calls, backend URLs, headers, generation parameters
print("--- Searching for Backend API routes and fetch patterns ---")
fetch_snippets = []
for m in re.finditer(r'fetch\([^\)]+\)', js):
    snippet = js[max(0, m.start() - 30):min(len(js), m.end() + 50)]
    fetch_snippets.append(snippet)

for s in fetch_snippets[:10]:
    print("Fetch call:", s)

print("\n--- Searching for 'generation' / 'llm' / 'model' / 'chunks' / 'corpus' / 'vector' ---")
for kw in ['generation', 'generator', 'model', 'llm', 'chunk', 'corpus', 'dataset', 'total', 'prompt', 'system']:
    matches = [m.start() for m in re.finditer(r'\b' + kw + r'\b', js, re.IGNORECASE)]
    print(f"Keyword '{kw}': {len(matches)} occurrences")
    for idx in matches[:2]:
        print(f"   [{kw}]:", js[max(0, idx - 40):min(len(js), idx + 80)].replace('\n', ' '))
