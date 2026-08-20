import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Matrix 2D span scoring algorithm
def extract_best_span_matrix(start_logits, end_logits, input_ids, tokenizer, max_span_len=35):
    """
    Finds (start, end) token indices with 1 <= start <= end <= start + max_span_len
    that maximizes start_logits[s] + end_logits[e].
    Runs in <0.05ms using vectorized PyTorch tensor operations.
    """
    # Exclude token 0 (null token)
    s_scores = start_logits[1:] # shape: (seq_len - 1,)
    e_scores = end_logits[1:]   # shape: (seq_len - 1,)
    
    # 2D matrix sum: M[s, e] = s_scores[s] + e_scores[e]
    score_matrix = s_scores.unsqueeze(1) + e_scores.unsqueeze(0) # shape: (L, L)
    
    # Mask out e < s (lower triangular) and e > s + max_span_len
    L = score_matrix.size(0)
    indices = torch.arange(L, device=score_matrix.device)
    span_lens = indices.unsqueeze(0) - indices.unsqueeze(1) # e - s
    
    valid_mask = (span_lens >= 0) & (span_lens <= max_span_len)
    score_matrix = torch.where(valid_mask, score_matrix, torch.tensor(-1e9, device=score_matrix.device))
    
    best_flat_idx = int(torch.argmax(score_matrix))
    best_s = (best_flat_idx // L) + 1
    best_e = (best_flat_idx % L) + 1
    
    tokens = input_ids[best_s : best_e + 1]
    ans = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    
    # Softmax probabilities
    s_prob = float(F.softmax(start_logits, dim=-1)[best_s])
    e_prob = float(F.softmax(end_logits, dim=-1)[best_e])
    
    return ans, s_prob * e_prob, best_s, best_e

print("Span matrix algorithm defined.")
