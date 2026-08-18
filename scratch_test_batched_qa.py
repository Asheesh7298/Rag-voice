import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import time

print("Testing batched QA logic...")
model_name = "deepset/xlm-roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

question = "दीमक बॉन्ड की कीमत कितनी होती है?"
contexts = [
    "दीमक बॉन्ड की कीमत आम तौर पर प्रति वर्ष $100 के आसपास होती है, लेकिन वे प्रति वर्ष $1000 तक जा सकते हैं। यह उपचार के प्रकार पर निर्भर करता है।",
    "दीमक के नुकसान की मरम्मत में हजारों डॉलर खर्च हो सकते हैं। नियमित निरीक्षण से रोकथाम संभव है।",
    "घरों में दीमक की रोकथाम के लिए विशेष रासायनिक उपचार किए जाते हैं।"
]

# Batched forward pass
t0 = time.perf_counter()
inputs = tokenizer(
    [question] * len(contexts),
    contexts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

s_logits = outputs.start_logits
e_logits = outputs.end_logits

s_probs = F.softmax(s_logits, dim=-1)
e_probs = F.softmax(e_logits, dim=-1)

best_score = -1.0
best_answer = ""
best_idx = 0

for i in range(len(contexts)):
    s_idx = int(torch.argmax(s_logits[i]))
    e_idx = int(torch.argmax(e_logits[i]))
    if e_idx < s_idx:
        e_idx = s_idx
    tokens = inputs["input_ids"][i][s_idx:e_idx + 1]
    ans = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    score = float(s_probs[i][s_idx] * e_probs[i][e_idx])
    print(f"Context {i+1}: score={score:.6f} | extracted={ans!r}")
    if score > best_score and ans:
        best_score = score
        best_answer = ans
        best_idx = i

elapsed_ms = (time.perf_counter() - t0) * 1000
print(f"\nBest Answer: {best_answer!r} (Score: {best_score:.6f})")
print(f"Batched Inference Latency on {device}: {elapsed_ms:.2f} ms")
