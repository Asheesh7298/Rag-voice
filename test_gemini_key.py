import os
import httpx
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
print("Testing Gemini API with key:", api_key[:10] + "...")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
body = {
    "contents": [{"role": "user", "parts": [{"text": "Answer in 1 line: What is the capital of India?"}]}]
}

try:
    with httpx.Client(timeout=10) as client:
        resp = client.post(url, json=body)
        print("Status code:", resp.status_code)
        print("Response:", resp.text[:300])
except Exception as e:
    print("Error:", e)
