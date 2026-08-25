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


def _clean_text_fragment(t: str) -> str:
    """Strips leading bullet numbers, breadcrumb trails, and dangling punctuation."""
    t = re.sub(r'^[0-9]+[\.\)\s\-]+', '', t.strip())
    t = re.sub(r'^[\>\#\*\-\s]+', '', t.strip())
    t = re.sub(r'[\.\,\;\:\s\-\>]+$', '.', t.strip())
    return t


def _check_intent_alignment(query: str, span: str, sentence: str) -> bool:
    """Verifies that the extracted answer span/sentence semantically satisfies the question intent."""
    q_lower = query.lower()
    span_lower = span.lower()
    full_lower = sentence.lower()

    # 1. Temporal: when, how long, how old, date, year
    if any(w in q_lower for w in ["when", "how long", "how old", "what year", "what time", "how many days", "how many years"]):
        time_terms = ["year", "years", "month", "months", "day", "days", "week", "weeks", "hour", "hours", 
                      "minute", "minutes", "second", "seconds", "century", "bc", "ad", "january",
                      "february", "march", "april", "may", "june", "july", "august", "september", "october",
                      "november", "december", "ago", "since", "until", "duration", "time", "date"]
        has_time = any(t in span_lower or t in full_lower for t in time_terms) or bool(re.search(r'\b\d{1,4}\b', span_lower))
        return has_time

    # 2. Quantitative: how many, how much, what percentage
    if any(w in q_lower for w in ["how many", "how much", "what percentage", "how high", "how fast", "how far"]):
        has_number = bool(re.search(r'\b\d+(\.\d+)?\b', span_lower)) or any(w in span_lower for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "percent", "%", "none", "zero"])
        return has_number

    # 3. Person / Who: who, whom, whose, creator, founder
    if any(w in q_lower for w in ["who is", "who was", "who are", "who were", "who created", "who founded", "whom"]):
        has_name = bool(re.search(r'\b[A-Z][a-z]+\b', span)) or any(w in span_lower for w in ["he", "she", "they", "king", "queen", "president", "author", "founder", "inventor", "doctor", "leader", "creator", "god"])
        return has_name

    # 4. Location / Where
    if any(w in q_lower for w in ["where is", "where was", "where are", "where were", "what city", "what country", "what state"]):
        has_loc = any(prep in full_lower for prep in [" in ", " at ", " near ", " located ", " city ", " country ", " state ", " county ", " north ", " south ", " east ", " west "]) or bool(re.search(r'\b[A-Z][a-z]+\b', span))
        return has_loc

    # 5. Definition / General
    return True


def _expand_to_sentence(span: str, context: str, query: str = "") -> str:
    """Expands a short extracted span to the most relevant, complete, clean sentence."""
    span_clean = span.strip()
    if len(span_clean) > 120:
        return _clean_text_fragment(span_clean)

    # Split context into clean sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n।])\s+', context) if s.strip()]
    matching = [s for s in sentences if span_clean in s]
    if not matching:
        matching = [s for s in sentences if span_clean.lower() in s.lower()]

    if matching:
        # If multiple sentences match the span, pick the one with highest query overlap
        if len(matching) > 1 and query:
            q_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 2)
            scored = []
            for s in matching:
                s_words = set(w.lower() for w in re.findall(r'\w+', s) if len(w) > 2)
                overlap = len(q_words & s_words)
                scored.append((overlap, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = scored[0][1]
        else:
            chosen = matching[0]

        cleaned = _clean_text_fragment(chosen)
        if len(cleaned) >= len(span_clean) and len(cleaned) < 320:
            if not cleaned.lower().startswith(("ways to", "how to", "step ")):
                return cleaned
            return cleaned

    return _clean_text_fragment(span_clean)



def generate_answer(query: str, results: Optional[List[Any]] = None) -> Answer:
    """
    Extracts an answer given a query and retrieved results context.
    Optimized for high correctness, low false confidence, and reliable refusal.
    """
    t0 = time.perf_counter()

    if results and len(results) > 0:
        try:
            # Retrieval score gating — only drop completely irrelevant passages (< 0.62)
            best_retrieval_score = max(
                (getattr(r, "score", 0.0) for r in results), default=0.0
            )
            if best_retrieval_score < 0.62:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return Answer(
                    text="Decline: The retrieved context does not contain sufficient information to answer this question.",
                    grounded=False, generation_ms=elapsed_ms
                )

            tokenizer, model = _get_qa_model()
            best_ans = ""
            best_prob = -1.0       # rank by composite score
            best_raw = -999.0
            best_grounded = False

            # evaluate top 5 chunks with full 512 capacity
            for r in results[:5]:
                context = getattr(r, "text", str(r)).strip()
                if not context or len(context) < 10:
                    continue

                inputs = tokenizer(query, context, return_tensors="pt", max_length=512, truncation=True)

                with torch.no_grad():
                    outputs = model(**inputs)
                    start_logits_raw = outputs.start_logits[0]
                    end_logits_raw = outputs.end_logits[0]

                    # Token 0 is <s> (SQuAD2 unanswerable indicator)
                    null_score = float(start_logits_raw[0].item() + end_logits_raw[0].item())

                    # Mask non-context tokens for span extraction
                    start_logits = start_logits_raw.clone()
                    end_logits = end_logits_raw.clone()
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

                        # Answer length & formatting sanity
                        if len(ans_span) < 3 or not any(c.isalnum() for c in ans_span):
                            continue

                        # Reject headline titles and company profiles
                        if any(p in ans_span.lower() for p in ["company profile", "activities association", "overview", "table of contents"]):
                            continue

                        expanded = _expand_to_sentence(ans_span, context, query)

                        # Filter incomplete sentences ending in dangling words (e.g. 'means.', 'is.', 'such as.')
                        clean_expanded = expanded.strip().rstrip(".").strip().lower()
                        if clean_expanded.endswith(("means", "is", "are", "was", "were", "such as", "that", "with", "of", "and", "or", "in", "to", "for", "by", "from")):
                            continue

                        # Filter personal forum posts & opinion statements
                        if expanded.lower().startswith(("i ", "i'm ", "i've ", "i would ", "i am ", "my ", "me ")):
                            continue

                        # Filter garbled text & OCR artifacts
                        if re.search(r'\.[a-z]{1,4}\s+[a-z]+', expanded):
                            expanded = _clean_text_fragment(ans_span)

                        aligned = _check_intent_alignment(query, ans_span, expanded)

                        # Require minimum probability floor (rejects garbled low-prob fragments)
                        if prob_score < 0.02 or raw_score < 2.0:
                            continue

                        # Dynamic Thresholding:
                        if aligned and prob_score > 0.02 and raw_score > 2.0:
                            is_valid = True
                        elif not aligned and prob_score > 0.18 and raw_score > 6.0:
                            is_valid = True
                        else:
                            is_valid = False

                        if not is_valid:
                            continue

                        # Query word analysis
                        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", 
                                     "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                                     "do", "does", "did", "the", "a", "an", "and", "or", "but", "in", "on", 
                                     "at", "to", "for", "with", "by", "about", "against", "between", "into", 
                                     "through", "during", "before", "after", "above", "below", "from", "up", 
                                     "down", "in", "out", "of", "off", "over", "under", "again", "further", 
                                     "then", "once", "here", "there", "all", "any", "both", "each", "few", 
                                     "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
                                     "own", "same", "so", "than", "too", "very", "can", "will", "just", 
                                     "should", "now", "tell", "describe", "explain", "meaning", "definition", "define"}

                        q_tokens = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
                        q_content_words = set(w for w in q_tokens if w not in stopwords)
                        c_words = set(w.lower() for w in re.findall(r'\w+', context) if len(w) > 2)
                        a_words = set(w.lower() for w in re.findall(r'\w+', expanded) if len(w) > 2)

                        is_definition = any(p in query.lower() for p in ["define", "definition", "what is", "meaning"])

                        # Context overlap check
                        if q_content_words:
                            ctx_overlap = len(q_content_words & c_words) / len(q_content_words)
                            ans_overlap = len(q_content_words & a_words) / len(q_content_words)
                        else:
                            ctx_overlap = 1.0
                            ans_overlap = 1.0

                        # Reject ungrounded guesses on irrelevant passages
                        if not is_definition and ctx_overlap < 0.15 and raw_score < 4.0:
                            continue

                        # Select the best candidate across all 5 chunks by cross-attention span score
                        if span_score > best_raw:
                            best_raw = span_score
                            best_prob = prob_score
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

