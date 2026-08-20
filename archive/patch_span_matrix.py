with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """            for i, chunk in enumerate(candidate_chunks):
                s_idx = int(torch.argmax(s_logits[i]))
                e_idx = int(torch.argmax(e_logits[i]))

                if e_idx < s_idx:
                    e_idx = s_idx

                tokens = inputs["input_ids"][i][s_idx:e_idx + 1]
                answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()

                # If model selected null token <s> (index 0) or decoded answer is empty, extract best non-null span
                if (not answer or s_idx == 0) and s_logits[i].shape[0] > 1:
                    s_idx_nz = int(torch.argmax(s_logits[i][1:])) + 1
                    e_idx_nz = int(torch.argmax(e_logits[i][s_idx_nz:])) + s_idx_nz
                    tokens_nz = inputs["input_ids"][i][s_idx_nz:e_idx_nz + 1]
                    ans_nz = self.qa_tokenizer.decode(tokens_nz, skip_special_tokens=True).strip()
                    if ans_nz and len(ans_nz) >= 2:
                        s_idx, e_idx = s_idx_nz, e_idx_nz
                        answer = ans_nz

                start_prob = float(s_probs[i][s_idx])
                end_prob = float(e_probs[i][e_idx])
                score = start_prob * end_prob

                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3
                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.25 * i)
                    score = round(score * rank_decay, 4)

                    if score > best_cand["score"]:
                        best_cand = {
                            "answer": answer,
                            "score": score,
                            "chunk_idx": i,
                            "source_text": chunk["text"],
                            "lang": chunk.get("lang"),
                        }"""

replacement = """            for i, chunk in enumerate(candidate_chunks):
                input_id_seq = inputs["input_ids"][i]
                seq_len = input_id_seq.shape[0]

                # Exact 2D matrix span search over bounded lengths (1 <= start <= end <= start + 35)
                if seq_len > 1:
                    s_sub = s_logits[i][1:]
                    e_sub = e_logits[i][1:]
                    L = s_sub.size(0)

                    score_matrix = s_sub.unsqueeze(1) + e_sub.unsqueeze(0)
                    indices = torch.arange(L, device=s_logits.device)
                    span_lens = indices.unsqueeze(0) - indices.unsqueeze(1)

                    valid_mask = (span_lens >= 0) & (span_lens <= 35)
                    score_matrix = torch.where(valid_mask, score_matrix, torch.tensor(-1e9, device=s_logits.device))

                    best_flat_idx = int(torch.argmax(score_matrix))
                    best_s = (best_flat_idx // L) + 1
                    best_e = (best_flat_idx % L) + 1

                    tokens = input_id_seq[best_s : best_e + 1]
                    answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()

                    start_prob = float(s_probs[i][best_s])
                    end_prob = float(e_probs[i][best_e])
                    score = start_prob * end_prob
                else:
                    answer = ""
                    score = 0.0

                if answer:
                    ans_lower = answer.lower()
                    if len(re.split(r'[,،]', answer)) >= 3:
                        score *= 0.5
                    if any(term in ans_lower for term in BIO_TERMS):
                        score *= 1.3
                    # Rank weighting: prioritize higher ranked retrieved passages
                    rank_decay = 1.0 / (1.0 + 0.25 * i)
                    score = round(score * rank_decay, 4)

                    if score > best_cand["score"]:
                        best_cand = {
                            "answer": answer,
                            "score": score,
                            "chunk_idx": i,
                            "source_text": chunk["text"],
                            "lang": chunk.get("lang"),
                        }"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully replaced with 2D span matrix search in modal_app.py!")
else:
    print("Target string not found, check matching.")
