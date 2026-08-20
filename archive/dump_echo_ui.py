import urllib.request, re

with open('archive/inspect_echo.py', 'r') as f:
    pass

req = urllib.request.Request('https://echo.omchillure.space/assets/index-B3OMOu1w.js', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    js = resp.read().decode('utf-8')

# Find all JSX / UI strings
strings = re.findall(r'`([^`]{3,120})`', js)
ui_strings = [s for s in strings if not s.startswith('http') and not s.startswith('rgba') and not s.startswith('var(') and any(c.isalpha() for c in s)]

print("Total UI strings found:", len(ui_strings))
print("\n--- Key UI Sections & Copy from https://echo.omchillure.space/ ---")
seen = set()
for s in ui_strings:
    if s not in seen and any(k in s.lower() for k in ['echo', 'rag', 'voice', 'guardrail', 'latency', 'sla', 'p50', 'p70', 'p100', 'benchmark', 'telemetry', 'stage', 'stt', 'grounding', 'indic', 'retriev', 'ms', 'goa', 'audit', 'source']):
        print(" •", s)
        seen.add(s)
