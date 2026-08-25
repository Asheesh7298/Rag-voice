#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Python Embedder for rag-local-eval-loop.
Interface matching: embed(texts), embed_one(text), get_model()
"""

import unicodedata
import re
from typing import List, Union
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _model

def _normalize_text(t: str) -> str:
    """Normalizes Unicode representation (NFC for Indic scripts) and cleans whitespace."""
    if not t:
        return ""
    t = unicodedata.normalize('NFC', t.strip())
    t = re.sub(r'\s+', ' ', t)
    return t

def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Computes 768-dim FP16 normalized embeddings for a list of passage texts.
    Uses 'passage: ' prefix for document corpus chunks as required by multilingual-e5-base.
    """
    if isinstance(texts, str):
        texts = [texts]
    formatted = [
        t if t.startswith(("passage: ", "query: ")) else f"passage: {_normalize_text(t)}"
        for t in texts
    ]
    model = get_model()
    embeddings = model.encode(formatted, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)

def embed_one(text: str) -> np.ndarray:
    """
    Embeds a single search query using 'query: ' prefix as required by multilingual-e5-base.
    """
    clean_text = _normalize_text(text)
    formatted = clean_text if clean_text.startswith(("query: ", "passage: ")) else f"query: {clean_text}"
    model = get_model()
    embeddings = model.encode([formatted], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings[0], dtype=np.float32)

def embed_query(query: str) -> List[float]:
    return embed_one(query).tolist()

def embed_passages(passages: List[str]) -> List[List[float]]:
    return embed(passages).tolist()


