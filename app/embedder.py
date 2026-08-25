#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Python Embedder for rag-local-eval-loop.
Interface matching: embed(texts), embed_one(text), get_model()
"""

from typing import List, Union
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _model

def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Computes 768-dim FP16 normalized embeddings for a list of texts.
    """
    if isinstance(texts, str):
        texts = [texts]
    formatted = [t if t.startswith(("passage: ", "query: ")) else f"query: {t}" for t in texts]
    model = get_model()
    embeddings = model.encode(formatted, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)

def embed_one(text: str) -> np.ndarray:
    """
    Embeds a single query or text string.
    """
    return embed([text])[0]

def embed_query(query: str) -> List[float]:
    return embed_one(query).tolist()

def embed_passages(passages: List[str]) -> List[List[float]]:
    formatted = [p if p.startswith("passage: ") else f"passage: {p}" for p in passages]
    model = get_model()
    embeddings = model.encode(formatted, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()
