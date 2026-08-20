import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Loading Qwen2.5-0.5B-Instruct on RTX 4050 GPU...")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

test_prompts = [
    "User: भारत की राजधानी क्या है? संक्षिप्त में 1 वाक्य में उत्तर दें।\nAssistant:",
    "User: भारताची राजधानी कोणती आहे? 1 वाक्यात उत्तर द्या.\nAssistant:",
    "User: What is photosynthesis? Answer in 1 clear sentence.\nAssistant:",
    "User: What are main symptoms of diabetes? Answer in 1 sentence.\nAssistant:",
    "User: वाघ कुठे राहतात? 1 वाक्यात उत्तर द्या.\nAssistant:"
]

print("=" * 80)
print("TESTING QWEN2.5-0.5B SLM ON RTX 4050 GPU")
print("=" * 80)

for p in test_prompts:
    inputs = tokenizer(p, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    lat = round((time.perf_counter() - t0) * 1000, 2)
    ans = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"\nPrompt: {p.splitlines()[0]}")
    print(f"Output: {ans}")
    print(f"Generation Latency: {lat} ms")
