with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _keyword_overlap_score to use char trigrams for Indic morphology matching
target1 = """    def _keyword_overlap_score(self, query_tokens: list, passage_text: str) -> float:
        \"\"\"
        Compute the ratio of query content words present in passage text.
        Case-insensitive string containment check.
        \"\"\"
        if not query_tokens:
            return 0.0
        p_lower = passage_text.lower()
        matches = sum(1 for token in query_tokens if token.lower() in p_lower)
        return matches / len(query_tokens)"""

replacement1 = """    def _keyword_overlap_score(self, query_tokens: list, passage_text: str) -> float:
        \"\"\"
        Compute morphological & lexical overlap using word containment and character trigrams.
        Handles inflected Indic root words accurately.
        \"\"\"
        if not query_tokens:
            return 0.0
        p_lower = passage_text.lower()
        word_matches = sum(1 for token in query_tokens if token.lower() in p_lower)
        word_ratio = word_matches / len(query_tokens)

        # Character trigrams for root-word matching
        q_str = " ".join(query_tokens).lower()
        if len(q_str) >= 3 and len(p_lower) >= 3:
            q_tri = set(q_str[i:i+3] for i in range(len(q_str)-2))
            p_tri = set(p_lower[i:i+3] for i in range(len(p_lower)-2))
            tri_ratio = len(q_tri & p_tri) / max(1, len(q_tri))
        else:
            tri_ratio = 0.0

        return max(word_ratio, tri_ratio)"""

# 2. Update _retrieve to keep rerank_n=50 and top_k=8 across all queries (including English)
target2 = """        is_eng = self._is_english_query(query)
        rerank_n = 20 if is_eng else self.RERANK_TOP_N
        top_k = 5 if is_eng else self.TOP_K"""

replacement2 = """        is_eng = self._is_english_query(query)
        rerank_n = self.RERANK_TOP_N
        top_k = self.TOP_K"""

if target1 in content and target2 in content:
    content = content.replace(target1, replacement1).replace(target2, replacement2)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched morphological overlap and retrieval depth in modal_app.py!")
else:
    print("Target strings not found, check matching...")
