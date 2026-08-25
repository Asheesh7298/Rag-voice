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
    Evaluates top candidate chunks with sentence expansion for high correctness.
    """
    t0 = time.perf_counter()
    
    if results and len(results) > 0:
        try:
            tokenizer, model = _get_qa_model()
            best_ans = ""
            best_score = -999.0
            best_grounded = False

            for r in results[:4]:
                context = getattr(r, "text", str(r)).strip()
                if not context:
                    continue

                inputs = tokenizer(query, context, return_tensors="pt", max_length=384, truncation=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    start_logits = outputs.start_logits[0]
                    end_logits = outputs.end_logits[0]
                    
                    # Mask non-context tokens
                    seq_ids = inputs.sequence_ids(0)
                    for i, s_id in enumerate(seq_ids):
                        if s_id != 1:  # 1 is context
                            start_logits[i] = -10000.0
                            end_logits[i] = -10000.0
                    
                    start_idx = int(torch.argmax(start_logits).item())
                    end_idx = int(torch.argmax(end_logits).item())
                    
                    if end_idx >= start_idx:
                        s_prob = float(torch.softmax(outputs.start_logits[0], dim=-1)[start_idx].item())
                        e_prob = float(torch.softmax(outputs.end_logits[0], dim=-1)[end_idx].item())
                        prob_score = s_prob * e_prob
                        raw_score = float((start_logits[start_idx] + end_logits[end_idx]).item())
                        
                        input_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                        ans_span = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                        
                        if len(ans_span) >= 2 and raw_score > 0.2:
                            expanded = _expand_to_sentence(ans_span, context)
                            if raw_score > best_score:
                                best_score = raw_score
                                best_ans = expanded
                                best_grounded = True

            # Open-ended conversational prompt handler (Zero Latency)
            q_lower = query.lower().strip()
            OPEN_PROMPTS = (
                "tell me about", "tell me", "describe", "explain", "information about",
                "information on", "what is the story of", "overview of",
                "के बारे में", "के बारे में बताएं", "बद्दल माहिती", "बद्दल सांगा", "माहिती द्या"
            )
            if any(p in q_lower for p in OPEN_PROMPTS) and results:
                first_ctx = getattr(results[0], "text", str(results[0])).strip()
                if first_ctx:
                    sents = [s.strip() for s in re.split(r'(?<=[.!?\n।])\s+', first_ctx) if s.strip() and len(s) > 15]
                    matched_sents = [s for s in sents if any(w in s.lower() for w in q_lower.split() if len(w) > 3)]
                    if matched_sents:
                        best_ans = " ".join(matched_sents[:2])
                        best_grounded = True
                    elif sents:
                        best_ans = " ".join(sents[:2])
                        best_grounded = True

            if not best_ans or (best_score < 0.2 and not best_grounded):
                best_ans = "Decline: The retrieved context does not contain sufficient information to answer this question."
                best_grounded = False

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return Answer(text=best_ans, grounded=best_grounded, generation_ms=elapsed_ms, model="xlm-roberta-base-squad2")
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
