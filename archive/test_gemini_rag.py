import os
import httpx
import json
import time
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")

SYSTEM_PROMPT = """You are a multilingual AI assistant for the Hacker House Goa Voice-RAG application.
You are given a user query in any Indic language or English, and optional retrieved passages from the IndicMSMARCO knowledge base.
Answer accurately and concisely (1-2 sentences) in the SAME language as the query.
If the retrieved context contains the answer, ground your response on it.
If the retrieved context does not have the answer, use your general knowledge to provide the correct fact."""

test_queries = [
    "भारत की राजधानी क्या है?",
    "भारताची राजधानी कोणती आहे?",
    "what is photosynthesis?",
    "what are symptoms of diabetes?",
    "what is a normal blood pressure reading?",
    "what is the capital of France?",
    "दीमक बॉन्ड की कीमत कितनी होती है?",
    "वाघ कुठे राहतात?",
    "ജെഡി പോരാളികളുടെ തരങ്ങൾ"
]

print("=" * 80)
print("TESTING RETRIEVAL + GEMINI FLASH GENERATION")
print("=" * 80)

for q in test_queries:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": f"User Query: {q}"}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 150}
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=body)
            lat = round((time.perf_counter() - t0) * 1000, 2)
            if resp.status_code == 200:
                ans = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"\nQ: {q}")
                print(f"A: {ans}")
                print(f"Latency: {lat} ms")
            else:
                print(f"Error {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print("Exception:", e)
