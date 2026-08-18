with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        # 5. Expand single character / subword fragment answers using source sentence
        words = answer.split()
        if len(words) == 1 and len(answer) <= 3 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 40:
                    answer = sent.strip()
                    break"""

replacement = """        # 5. Expand extracted answer spans to the full informative sentence
        words = answer.split()
        if len(words) <= 12 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 45:
                    answer = sent.strip()
                    break"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched sentence expansion in modal_app.py!")
else:
    print("Target string not found!")
