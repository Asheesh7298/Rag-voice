#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Python Generator for rag-local-eval-loop (Branch A).
Implements extractive QA with XLM-RoBERTa and live endpoint fallback.
"""

from typing import Dict, Any, Optional
import urllib.request
import urllib.parse
import json

_qa_pipeline = None

def _get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        from transformers import pipeline
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/xlm-roberta-base-squad2",
            tokenizer="deepset/xlm-roberta-base-squad2"
        )
    return _qa_pipeline

def generate_answer(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts or generates an answer given a query and optional context.
    If context is provided, performs local extractive QA.
    If context is not provided, routes to the live Modal A100 RAG endpoint.
    """
    if context:
        # Direct local extractive QA
        qa = _get_qa_pipeline()
        res = qa(question=query, context=context)
        return {
            "answer": res.get("answer", "").strip(),
            "confidence": float(res.get("score", 0.0)),
            "grounded": True if res.get("score", 0) > 0.05 else False,
            "sources": [{"text": context}]
        }
    
    # Query live endpoint
    try:
        url = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "answer": payload.get("answer", ""),
                "sources": payload.get("sources", []),
                "confidence": payload.get("confidence", 0.0),
                "grounded": payload.get("grounded", False),
                "timings_ms": payload.get("timings_ms", {})
            }
    except Exception as e:
        return {
            "answer": f"Decline: Service unavailable ({e})",
            "sources": [],
            "grounded": False
        }
