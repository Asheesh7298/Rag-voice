with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3
                    score = round(score, 4)"""

replacement = """                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3
                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.25 * i)
                    score = round(score * rank_decay, 4)"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched candidate scoring with rank decay!")
else:
    print("Target not found!")
