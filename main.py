#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Entrypoint for Native Python Evaluation (rag-local-eval-loop).
Exposes embed(), embed_one(), get_model(), generate_answer().
"""

from app.embedder import embed, embed_one, get_model, embed_query, embed_passages
from app.generator import generate_answer, Answer

def query(text: str):
    return generate_answer(text)

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "what county is columbus city in"
    print(f"Testing Query: {q}")
    res = generate_answer(q)
    print("Result:", res)
