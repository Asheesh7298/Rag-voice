import urllib.request, json

def post_query(q):
    req = urllib.request.Request(
        'https://echo.omchillure.space/api/ask',
        data=json.dumps({"query": q}).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

test_queries = [
    # 1. English standard query
    "what county is columbus city in",
    # 2. English semantic paraphrase (where MiniLM dense should help vs BM25)
    "cost to replace vehicle alternator",
    # 3. Hindi query (Indic Devanagari)
    "फ्लोरिडा में गोल्फ कार्ट चलाने की उम्र क्या है",
    # 4. Hindi semantic query
    "ब्राइटन टाउनशिप फोन नंबर",
    # 5. Marathi query (Indic Devanagari)
    "फ्रान्सचे सध्याचे चलन काय आहे"
]

print("=== REVERSE-ENGINEERING ECHO BACKEND API ===")
for q in test_queries:
    res = post_query(q)
    print(f"\nQUERY: {q}")
    print(f"  Answer: {res.get('answer')}")
    print(f"  Total ms: {res.get('total_ms')}")
    print(f"  SLA ok: {res.get('sla_ok')}")
    
    hits = res.get('hits', [])
    print(f"  Hits returned: {len(hits)}")
    origins = [h.get('origin') for h in hits]
    strategies = [h.get('chunk', {}).get('strategy') for h in hits]
    languages = [h.get('chunk', {}).get('language') for h in hits]
    print(f"  Origins in hits: {origins}")
    print(f"  Strategies: {set(strategies)}")
    print(f"  Languages of hits: {languages[:5]}")
    if hits:
        top_hit = hits[0]
        print(f"  Top hit text: {top_hit.get('chunk', {}).get('text', '')[:120]}...")
        print(f"  Top hit score: {top_hit.get('score')}")
