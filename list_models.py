import os
import httpx
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
try:
    with httpx.Client(timeout=10) as client:
        resp = client.post(url) if False else client.get(url)
        print("Status:", resp.status_code)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            print(f"Found {len(models)} models:")
            for m in models:
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    print("  ", m['name'])
        else:
            print("Error:", resp.text[:300])
except Exception as e:
    print("Exception:", e)
