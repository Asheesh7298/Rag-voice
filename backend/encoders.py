"""
The two neural models, run with ONNX Runtime (no PyTorch):

- E5Encoder: multilingual-e5-base sentence embeddings (question -> 768-dim unit vector).
- QAReader:  xlm-roberta-base-squad2 extractive QA (question + passages -> start/end scores per token).

Both reproduce what v1 computed with sentence-transformers / transformers 4.44: the same tokenizer
files, truncation and padding, the same pooling, and full FP32 math on the GPU (TF32 switched off).
"""
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
from tokenizers import Encoding, Tokenizer


def make_session(model_path: str) -> ort.InferenceSession:
    """GPU session when CUDA is available, else CPU."""
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, ("CUDAExecutionProvider", {
            "device_id": 0,
            "use_tf32": 0,                                # PyTorch's default: full FP32 matmuls
            "arena_extend_strategy": "kSameAsRequested",  # grow GPU memory only as needed (vectors use ~20 GB)
        }))
    return ort.InferenceSession(model_path, providers=providers)


def _clean_up_tokenization(text: str) -> str:
    """transformers' `clean_up_tokenization`, which `decode()` applies when clean_up_tokenization_spaces=True."""
    return (text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
            .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s")
            .replace(" 've", "'ve").replace(" 're", "'re"))


class E5Encoder:
    def __init__(self, model_dir: str, max_length: int = 64):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.no_padding()
        self.session = make_session(f"{model_dir}/model.onnx")
        self.input_names = {i.name for i in self.session.get_inputs()}
        outputs = [o.name for o in self.session.get_outputs()]
        self.output_name = "last_hidden_state" if "last_hidden_state" in outputs else outputs[0]

    def embed(self, text: str) -> np.ndarray:
        """Mean of the token vectors (ignoring padding), L2-normalized, as sentence-transformers does."""
        enc = self.tokenizer.encode(text.strip())      # sentence-transformers strips input text
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run([self.output_name], feeds)[0]            # [1, tokens, 768]
        weights = mask[..., None].astype(np.float32)
        vec = ((hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None))[0]
        return (vec / np.linalg.norm(vec)).astype(np.float32)


class QAReader:
    def __init__(self, model_dir: str, max_length: int = 256):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        pad_token = "<pad>"
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad_token), pad_token=pad_token)
        with open(f"{model_dir}/tokenizer_config.json", encoding="utf-8") as f:
            self.clean_up_spaces = json.load(f).get("clean_up_tokenization_spaces", True)
        self.session = make_session(f"{model_dir}/model.onnx")

    def encode_pairs(self, questions: list[str], contexts: list[str]) -> list[Encoding]:
        """Tokenize (question, passage) pairs, padded to the longest pair in the batch."""
        return self.tokenizer.encode_batch(list(zip(questions, contexts)))

    def logits(self, encodings: list[Encoding]) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass over the batch: start and end scores per token, each [batch, tokens] float32."""
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        start, end = self.session.run(["start_logits", "end_logits"], {"input_ids": ids, "attention_mask": mask})
        return start.astype(np.float32), end.astype(np.float32)

    def decode(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return _clean_up_tokenization(text) if self.clean_up_spaces else text
