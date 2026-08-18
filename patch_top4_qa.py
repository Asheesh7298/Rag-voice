with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update candidate batch size from 3 to 4 in _extract_best_answer
target1 = """        is_eng = self._is_english_query(question)
        if is_eng:
            english_chunks = [c for c in valid_chunks if self._is_english_query(c.get("text", ""))]
            if english_chunks:
                best = _evaluate_batched_chunks(english_chunks[:3])
            else:
                best = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": "", "lang": None}

            if best["score"] <= 0.05 or not best["answer"]:
                fallback_best = _evaluate_batched_chunks(valid_chunks[:3])
                if fallback_best["score"] > best["score"] and fallback_best["answer"]:
                    best = fallback_best
        else:
            best = _evaluate_batched_chunks(valid_chunks[:3])"""

replacement1 = """        is_eng = self._is_english_query(question)
        if is_eng:
            english_chunks = [c for c in valid_chunks if self._is_english_query(c.get("text", ""))]
            if english_chunks:
                best = _evaluate_batched_chunks(english_chunks[:4])
            else:
                best = {"answer": "", "score": -1.0, "chunk_idx": 0, "source_text": "", "lang": None}

            if best["score"] <= 0.05 or not best["answer"]:
                fallback_best = _evaluate_batched_chunks(valid_chunks[:4])
                if fallback_best["score"] > best["score"] and fallback_best["answer"]:
                    best = fallback_best
        else:
            best = _evaluate_batched_chunks(valid_chunks[:4])"""

# 2. Add question-answer entity/keyword alignment boost in 2D span scoring
target2 = """                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3
                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.25 * i)
                    score = round(score * rank_decay, 4)"""

replacement2 = """                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3

                    # Keyword / entity relevance bonus
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

if target1 in content and target2 in content:
    content = content.replace(target1, replacement1).replace(target2, replacement2)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched modal_app.py with top-4 candidate batching and keyword relevance bonus!")
else:
    print("Target strings not found, checking...")
