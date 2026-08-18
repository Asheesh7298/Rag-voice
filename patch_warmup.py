with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'self._extract_best_answer("warmup question", [{"text": "warmup context for the model"}])'
replacement = 'self._extract_best_answer("warmup question", [{"text": "warmup context 1"}, {"text": "warmup context 2"}, {"text": "warmup context 3"}, {"text": "warmup context 4"}])'

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched batch-4 warmup in modal_app.py!")
else:
    print("Target not found!")
