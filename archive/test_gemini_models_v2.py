import os
import httpx
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")

for model in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.6-flash"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": "What is the capital of India? Reply in 5 words."}]}]
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=body)
            print(f"Model {model}: status {resp.status_code}")
            if resp.status_code == 200:
                print("  Answer:", resp.json()['candidates'][0]['content']['parts'][0]['text'].strip())
            else:
                print("  Error:", resp.text[:150])
    except Exception as e:
        print(f"Model {model}: exception {e}")
