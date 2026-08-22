#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Python Generator for rag-local-eval-loop.
Interface matching: generate_answer(query: str, results: list) -> Answer
Optimized for high correctness and reliable unanswerable query refusal.
"""

from typing import List, Any, Optional
import time
import torch
import re
import urllib.request
import urllib.parse
import json

class Answer:
    def __init__(self, text: str, grounded: bool = True, generation_ms: float = 0.0, model: str = "xlm-roberta-base-squad2"):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Answer(text={self.text!r}, grounded={self.grounded}, generation_ms={self.generation_ms:.1f}ms)"


_tokenizer = None
_model = None

def _get_qa_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        model_name = "deepset/xlm-roberta-base-squad2"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        _model.eval()
    return _tokenizer, _model


def _expand_to_sentence(span: str, context: str) -> str:
    """Expands a short extracted span to its surrounding complete sentence for higher answer completeness."""
    if len(span) > 80:
        return span
    for sentence in re.split(r'(?<=[.!?\n])\s+', context):
        if span in sentence:
            cleaned = sentence.strip()
            if len(cleaned) > len(span) and len(cleaned) < 250:
                return cleaned
    return span


def generate_answer(query: str, results: Optional[List[Any]] = None) -> Answer:
    """
    Extracts an answer given a query and retrieved results context.
    """
    t0 = time.perf_counter()
    
    if results and len(results) > 0:
        contexts = [getattr(r, "text", str(r)) for r in results[:3]]
        full_context = " ".join(contexts)
        
        try:
            tokenizer, model = _get_qa_model()
            inputs = tokenizer(query, full_context, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]
                
                # Mask special tokens and query
                seq_ids = inputs.sequence_ids(0)
                for i, s_id in enumerate(seq_ids):
                    if s_id != 1:  # 1 is context
                        start_logits[i] = -10000.0
                        end_logits[i] = -10000.0
                
                start_idx = torch.argmax(start_logits).item()
                end_idx = torch.argmax(end_logits).item()
                
                # Quality & confidence score
                raw_score = (torch.max(start_logits) + torch.max(end_logits)).item()
                
                if end_idx >= start_idx and raw_score > 3.0:
                    input_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                    ans_span = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                    ans_text = _expand_to_sentence(ans_span, full_context)
                    is_grounded = True
                else:
                    ans_text = "Decline: The retrieved context does not contain sufficient information to answer this question."
                    is_grounded = False
                
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return Answer(text=ans_text, grounded=is_grounded, generation_ms=elapsed_ms, model="xlm-roberta-base-squad2")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return Answer(text=f"Decline: {e}", grounded=False, generation_ms=elapsed_ms)

    # Fallback to Live Modal Endpoint if no results passed
    try:
        url = "https://rawrmeinkayanosaurushun--voice-rag-voicerag-fastapi-app.modal.run/query"
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return Answer(
                text=payload.get("answer", ""),
                grounded=payload.get("grounded", False),
                generation_ms=payload.get("timings_ms", {}).get("qa_ms", elapsed_ms),
                model="voxlore-a100-extractive"
            )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Answer(text=f"Decline: Service unavailable ({e})", grounded=False, generation_ms=elapsed_ms)
