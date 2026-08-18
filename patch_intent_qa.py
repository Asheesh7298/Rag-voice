with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """                    # Keyword / entity relevance bonus
                    q_words = set(re.findall(r'\\w+', question.lower()))
                    ans_words = set(re.findall(r'\\w+', ans_lower))
                    # Avoid trivial answers that just repeat the full question
                    if len(ans_words) > 0 and ans_words == q_words:
                        score *= 0.1
                    elif any(w in ans_words for w in q_words if len(w) > 3):
                        score *= 1.25

                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.15 * i)
                    score = round(score * rank_decay, 4)"""

replacement = """                    # Keyword / entity relevance bonus
                    q_words = set(re.findall(r'\\w+', question.lower()))
                    ans_words = set(re.findall(r'\\w+', ans_lower))
                    # Avoid trivial answers that just repeat the full question
                    if len(ans_words) > 0 and ans_words == q_words:
                        score *= 0.1
                    elif any(w in ans_words for w in q_words if len(w) > 3):
                        score *= 1.25

                    # Intent-specific entity bonus
                    # 1. Location intent (where / कहाँ / कुठे / কোথায় / ఎక్కడ / எங்கு)
                    if any(w in question.lower() for w in ("where", "कहाँ", "कुठे", "কোথায়", "ఎక్కడ", "எங்கு", "ਕਿੱਥੇ")):
                        if any(term in ans_lower for term in ("जंगल", "देश", "प्रदेश", "forest", "mountain", "country", "city", "क्षेत्र", "प्रदेशात", "मध्ये", "இல்")):
                            score *= 1.35

                    # 2. Cost / numerical intent (cost / price / how much / कितना / खर्च / কত / ధర / விலை)
                    if any(w in question.lower() for w in ("cost", "price", "how much", "कितना", "खर्च", "दर", "दाम", "কত", "ధర", "விலை")):
                        if any(c.isdigit() for c in answer) or any(s in answer for s in ("$", "₹", "€", "£")):
                            score *= 1.40

                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.15 * i)
                    score = round(score * rank_decay, 4)"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched intent-specific entity scoring in modal_app.py!")
else:
    print("Target string not found, check matching...")
