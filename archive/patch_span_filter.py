with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update 2D matrix span search to find the best span with len(answer.strip()) >= 4
target1 = """                    best_flat_idx = int(torch.argmax(score_matrix))
                    best_s = (best_flat_idx // L) + 1
                    best_e = (best_flat_idx % L) + 1

                    tokens = input_id_seq[best_s : best_e + 1]
                    answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()"""

replacement1 = """                    # Sort top candidates from score matrix and pick the best non-trivial span (length >= 4 chars)
                    flat_sorted = torch.argsort(score_matrix.view(-1), descending=True)
                    answer = ""
                    best_s, best_e = 1, 1
                    for flat_idx in flat_sorted[:10]:
                        s_cand = int(flat_idx // L) + 1
                        e_cand = int(flat_idx % L) + 1
                        toks = input_id_seq[s_cand : e_cand + 1]
                        ans_cand = self.qa_tokenizer.decode(toks, skip_special_tokens=True).strip()
                        if len(ans_cand) >= 4:
                            answer = ans_cand
                            best_s, best_e = s_cand, e_cand
                            break
                    if not answer:
                        best_flat_idx = int(torch.argmax(score_matrix))
                        best_s = (best_flat_idx // L) + 1
                        best_e = (best_flat_idx % L) + 1
                        tokens = input_id_seq[best_s : best_e + 1]
                        answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()"""

if target1 in content:
    content = content.replace(target1, replacement1)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched non-trivial span extraction in modal_app.py!")
else:
    print("Target string not found, check matching.")
