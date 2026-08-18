with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        # 5. Expand very short answers (<4 words) using source sentence
        words = answer.split()
        if len(words) < 4 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 40:
                    answer = sent.strip()
                    break"""

replacement = """        # 5. Expand single character / subword fragment answers using source sentence
        words = answer.split()
        if len(words) == 1 and len(answer) <= 3 and source_text and answer in source_text:
            sentences = re.split(r'(?<=[।.!?])\s+', source_text)
            for sent in sentences:
                if answer in sent and 3 <= len(sent.split()) <= 40:
                    answer = sent.strip()
                    break"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched concise direct answer preservation in modal_app.py!")
else:
    print("Target string not found, check matching.")
