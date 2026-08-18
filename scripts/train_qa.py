"""
Fine-tune XLM-RoBERTa on IndicMSMARCO for Extractive QA with SQuAD 2.0 Null-Score Support.

Data Source: /index/passages.jsonl on Modal Volume 'voice-rag-index' (12,999 passages across 13 Indic languages).
Base Model: deepset/xlm-roberta-base-squad2
Output: /index/qa-model-finetuned on the Modal Volume

Run:
  python -m modal run scripts/train_qa.py
"""
import modal
import os
import json
import random
import re

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.44.0",
        "torch>=2.1.0",
        "datasets>=2.19.0",
        "accelerate>=0.29.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "tqdm>=4.66.4",
        "numpy>=1.26,<3.0",
    )
    .run_commands(
        "python -c \""
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; "
        "AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model'); "
        "AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2').save_pretrained('/models/qa-model')"
        "\""
    )
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-train-qa", image=train_image)


def find_best_extractive_span(passage: str, answer: str):
    """
    Find the best matching character span (char_start, char_end) in passage for an answer.
    Uses exact match first, then word-level sliding window token F1.
    """
    if not passage or not answer:
        return None, 0.0

    # 1. Exact match
    idx = passage.find(answer)
    if idx != -1:
        return (idx, idx + len(answer)), 1.0

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    ans_tokens = tokenize(answer)
    if not ans_tokens:
        return None, 0.0
    ans_set = set(ans_tokens)

    words_with_offsets = []
    for m in re.finditer(r'\S+', passage):
        words_with_offsets.append((m.group(), m.start(), m.end()))

    if not words_with_offsets:
        return None, 0.0

    best_span = None
    best_f1 = 0.0
    n_words = len(words_with_offsets)
    ans_len = len(ans_tokens)

    min_w = max(1, int(ans_len * 0.4))
    max_w = min(n_words, int(ans_len * 1.8) + 5)

    for w_len in range(min_w, max_w + 1):
        for i in range(0, n_words - w_len + 1):
            window_words = words_with_offsets[i : i + w_len]
            span_text = passage[window_words[0][1] : window_words[-1][2]]
            span_tokens = tokenize(span_text)
            if not span_tokens:
                continue

            common = len(set(span_tokens) & ans_set)
            if common == 0:
                continue

            prec = common / len(span_tokens)
            rec = common / len(ans_set)
            f1 = 2 * prec * rec / (prec + rec)

            if f1 > best_f1:
                best_f1 = f1
                best_span = (window_words[0][1], window_words[-1][2])
                if f1 >= 0.95:
                    return best_span, best_f1

    return best_span, best_f1


@app.cls(
    gpu="A10G",
    volumes={"/index": volume},
    timeout=3600,
)
class IndicQATrainer:

    @modal.method()
    def train(self, num_epochs: int = 3, batch_size: int = 16, lr: float = 3e-5):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            TrainingArguments,
            Trainer,
            default_data_collator,
        )
        from datasets import Dataset

        print("=== Step 1: Loading Passages from Volume ===")
        passages = []
        with open("/index/passages.jsonl", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
        print(f"Loaded {len(passages)} total passages.")

        print("=== Step 2: Preparing SQuAD 2.0 Dataset ===")
        examples = []
        positive_count = 0
        negative_count = 0

        # Create answerable examples
        for p in passages:
            q = p.get("query", "").strip()
            text = p.get("text", "").strip()
            answers = p.get("answers", [])
            is_selected = p.get("is_selected", False)

            if not q or not text:
                continue

            if answers and answers[0] and is_selected:
                gold_ans = answers[0].strip()
                span, f1 = find_best_extractive_span(text, gold_ans)
                if span and f1 >= 0.35:
                    char_start, char_end = span
                    span_text = text[char_start:char_end]
                    examples.append({
                        "id": str(p.get("id", len(examples))),
                        "title": str(p.get("lang", "indic")),
                        "context": text,
                        "question": q,
                        "answers": {
                            "text": [span_text],
                            "answer_start": [char_start],
                        }
                    })
                    positive_count += 1

        # Create unanswerable (null-score) examples from unselected passages
        random.seed(42)
        unselected_passages = [p for p in passages if not p.get("is_selected", False)]
        if len(unselected_passages) < positive_count // 3:
            unselected_passages = passages

        # Sample unanswerable pairs (ratio ~ 1:3 unanswerable to answerable)
        target_negatives = int(positive_count * 0.30)
        sampled_negatives = random.sample(unselected_passages, min(target_negatives, len(unselected_passages)))
        for p in sampled_negatives:
            q = p.get("query", "").strip()
            text = p.get("text", "").strip()
            if q and text:
                examples.append({
                    "id": f"neg-{p.get('id', len(examples))}",
                    "title": str(p.get("lang", "indic")),
                    "context": text,
                    "question": q,
                    "answers": {
                        "text": [],
                        "answer_start": [],
                    }
                })
                negative_count += 1

        print(f"Total dataset size: {len(examples)} ({positive_count} answerable, {negative_count} unanswerable)")

        # Split train / val
        random.shuffle(examples)
        split_idx = int(len(examples) * 0.90)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        print(f"Train split: {len(train_examples)}, Val split: {len(val_examples)}")

        train_ds = Dataset.from_list(train_examples)
        val_ds = Dataset.from_list(val_examples)

        print("=== Step 3: Tokenizing with SQuAD 2.0 Offsets ===")
        model_path = "/models/qa-model" if os.path.exists("/models/qa-model") else "deepset/xlm-roberta-base-squad2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        max_seq_length = 384

        def preprocess_squad(batch):
            questions = [q.strip() for q in batch["question"]]
            inputs = tokenizer(
                questions,
                batch["context"],
                max_length=max_seq_length,
                truncation=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = batch["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]

                # Unanswerable
                if not answer["answer_start"] or not answer["text"]:
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find token bounds of context (sequence_id == 1)
                idx = 0
                while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if context_start > context_end or offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    token_start = context_start
                    while token_start <= context_end and offset[token_start][0] <= start_char:
                        token_start += 1
                    token_start -= 1

                    token_end = context_end
                    while token_end >= context_start and offset[token_end][1] >= end_char:
                        token_end -= 1
                    token_end += 1

                    start_positions.append(token_start)
                    end_positions.append(token_end)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        tokenized_train = train_ds.map(
            preprocess_squad,
            batched=True,
            remove_columns=train_ds.column_names,
            desc="Tokenizing train set",
        )
        tokenized_val = val_ds.map(
            preprocess_squad,
            batched=True,
            remove_columns=val_ds.column_names,
            desc="Tokenizing val set",
        )

        print("=== Step 4: Loading Base Model & Training ===")
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        training_args = TrainingArguments(
            output_dir="/tmp/qa_checkpoints",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=True,
            logging_steps=25,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        trainer.train()

        print("=== Step 5: Saving Fine-Tuned Model to Modal Volume ===")
        out_dir = "/index/qa-model-finetuned"
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Commit volume changes
        volume.commit()
        print(f"✅ Successfully trained and saved fine-tuned QA model to {out_dir} on Volume!")


@app.local_entrypoint()
def main(epochs: int = 3, batch_size: int = 16, lr: float = 3e-5):
    print("Launching IndicQATrainer.train on Modal GPU (A10G)...")
    trainer = IndicQATrainer()
    trainer.train.remote(num_epochs=epochs, batch_size=batch_size, lr=lr)
    print("Fine-tuning completed successfully!")
