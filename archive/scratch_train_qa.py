import modal
import json
import os
import random
import sys

# Define image with huggingface datasets and accelerate packages for training
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
)

volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-train", image=train_image)

@app.cls(
    gpu="A10G",  # Use A10G for fast training
    volumes={"/index": volume},
    timeout=1800,  # 30 mins limit
)
class TrainQA:
    
    @modal.method()
    def align_and_train(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        
        # 1. Loading pre-trained XLM-R SQuAD2 model
        print("Loading pre-trained model for pseudo-labeling...")
        model_name = "deepset/xlm-roberta-base-squad2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        # 2. Load passages
        print("Loading passages from volume...")
        passages = []
        with open("/index/passages.jsonl", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
        print(f"Loaded {len(passages)} passages.")
        
        # Helper to compute Token F1 between two strings
        def compute_f1(pred, gold):
            def normalize(t):
                return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()
            pred_toks = normalize(pred).split()
            gold_toks = normalize(gold).split()
            common = set(pred_toks) & set(gold_toks)
            if not pred_toks or not gold_toks:
                return 1.0 if pred_toks == gold_toks else 0.0
            prec = len(common) / len(pred_toks)
            rec = len(common) / len(gold_toks)
            return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            
        # 3. Perform pseudo-labeling
        print("Starting pseudo-labeling process...")
        labeled_data = []
        
        # Run inference in batches for speed
        batch_size = 64
        for idx in range(0, len(passages), batch_size):
            batch = passages[idx:idx + batch_size]
            questions = [p["query"] for p in batch]
            contexts = [p["text"] for p in batch]
            
            inputs = tokenizer(
                questions, contexts,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding=True,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model(**inputs)
                
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()
            input_ids = inputs["input_ids"].cpu()
            
            for i, p in enumerate(batch):
                s_logits = start_logits[i]
                e_logits = end_logits[i]
                
                start_idx = int(torch.argmax(s_logits))
                end_idx = int(torch.argmax(e_logits))
                
                if end_idx < start_idx:
                    end_idx = start_idx
                    
                tokens = input_ids[i][start_idx:end_idx + 1]
                pred_answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                
                best_f1 = 0.0
                for gold in p["answers"]:
                    f1 = compute_f1(pred_answer, gold)
                    if f1 > best_f1:
                        best_f1 = f1
                        
                if best_f1 >= 0.3 and pred_answer:
                    passage_text = p["text"]
                    char_start = passage_text.find(pred_answer)
                    if char_start != -1:
                        labeled_data.append({
                            "context": passage_text,
                            "question": p["query"],
                            "answers": {
                                "text": [pred_answer],
                                "answer_start": [char_start]
                            }
                        })
                        
        print(f"Successfully aligned {len(labeled_data)} / {len(passages)} passages ({len(labeled_data)/len(passages):.2%}).")
        
        if len(labeled_data) < 100:
            print("❌ Too few aligned passages. Aborting fine-tuning.")
            return
            
        # 4. Prepare SQuAD Dataset
        random.seed(42)
        random.shuffle(labeled_data)
        split_idx = int(len(labeled_data) * 0.9)
        train_list = labeled_data[:split_idx]
        val_list = labeled_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_list)
        val_dataset = Dataset.from_list(val_list)
        
        def preprocess_function(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=384,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        print("Preprocessing dataset for training...")
        tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
        
        # 5. Fine-tune model
        print("Starting HuggingFace Trainer fine-tuning...")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            fp16=True,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            report_to="none"
        )
        
        model.train()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        print("Training completed! Saving model weights to Volume at /index/qa-model-finetuned...")
        
        out_dir = "/index/qa-model-finetuned"
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        print("✅ Fine-tuning completed and saved successfully!")

@app.local_entrypoint()
def main():
    trainer = TrainQA()
    print("Launching TrainQA.align_and_train remote task...")
    trainer.align_and_train.remote()
    print("TrainQA remote task finished.")
