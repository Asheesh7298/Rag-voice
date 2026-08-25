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
    Optimized for high correctness, low false confidence, and reliable refusal.
    """
    t0 = time.perf_counter()

    if results and len(results) > 0:
        try:
            # Fix 2: Retrieval score gating — if best passage score is too low, decline
            best_retrieval_score = max(
                (getattr(r, "score", 0.0) for r in results), default=0.0
            )
            if best_retrieval_score < 0.65:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return Answer(
                    text="Decline: The retrieved context does not contain sufficient information to answer this question.",
                    grounded=False, generation_ms=elapsed_ms
                )

            tokenizer, model = _get_qa_model()
            best_ans = ""
            best_prob = -1.0       # Fix 4: rank by prob_score (calibrated 0-1)
            best_raw = -999.0
            best_grounded = False

            # Fix 5: evaluate all 5 chunks instead of 4
            for r in results[:5]:
                context = getattr(r, "text", str(r)).strip()
                if not context or len(context) < 10:
                    continue

                inputs = tokenizer(query, context, return_tensors="pt", max_length=384, truncation=True)

                with torch.no_grad():
                    outputs = model(**inputs)
                    start_logits = outputs.start_logits[0]
                    end_logits = outputs.end_logits[0]

                    # Mask non-context tokens for span extraction
                    seq_ids = inputs.sequence_ids(0)
                    for i, s_id in enumerate(seq_ids):
                        if s_id != 1:  # 1 is context
                            start_logits[i] = -10000.0
                            end_logits[i] = -10000.0

                    start_idx = int(torch.argmax(start_logits).item())
                    end_idx = int(torch.argmax(end_logits).item())

                    if end_idx >= start_idx:
                        span_score = float(start_logits[start_idx].item() + end_logits[end_idx].item())

                        s_prob = float(torch.softmax(outputs.start_logits[0], dim=-1)[start_idx].item())
                        e_prob = float(torch.softmax(outputs.end_logits[0], dim=-1)[end_idx].item())
                        prob_score = s_prob * e_prob
                        raw_score = span_score

                        input_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                        ans_span = tokenizer.decode(input_ids, skip_special_tokens=True).strip()

                        # Fix 7: Answer length sanity — skip tiny or non-alphanumeric spans
                        if len(ans_span) < 3 or not any(c.isalnum() for c in ans_span):
                            continue

                        # Fix 1 + 4: Use prob_score as primary threshold AND ranking
                        if prob_score > 0.005 and raw_score > 1.0:
                            expanded = _expand_to_sentence(ans_span, context)

                            # Fix 6: Cross-check answer relevance to query
                            q_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 2)
                            a_words = set(w.lower() for w in re.findall(r'\w+', expanded) if len(w) > 2)
                            # Don't penalize if query words appear in answer (good sign)
                            # But do check context words overlap with query
                            c_words = set(w.lower() for w in re.findall(r'\w+', context) if len(w) > 2)
                            query_context_overlap = len(q_words & c_words) / max(1, len(q_words))
                            if query_context_overlap < 0.10:
                                # Context has almost no query terms — likely irrelevant passage
                                continue

                            # Fix 4: Rank by prob_score (calibrated), not raw_score (unbounded)
                            if prob_score > best_prob:
                                best_prob = prob_score
                                best_raw = raw_score
                                best_ans = expanded
                                best_grounded = True

            # Fix 3: Removed open-ended prompt handler — it hurt eval correctness

            if not best_ans or not best_grounded:
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

