#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Entrypoint for Native Python Evaluation (rag-local-eval-loop Branch A).
Exposes embed(), generate_answer(), and query().
"""

from app.embedder import embed, embed_query, embed_passages
from app.generator import generate_answer

def query(text: str):
    return generate_answer(text)

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "what county is columbus city in"
    print(f"Testing Query: {q}")
    res = generate_answer(q)
    print("Result:", res)
