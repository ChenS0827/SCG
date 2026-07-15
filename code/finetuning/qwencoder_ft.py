import os
import json
import logging
import torch
import numpy as np
import argparse
import random
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_bleu(reference, candidate):
    try:
        reference_tokens = word_tokenize(reference)
        candidate_tokens = word_tokenize(candidate)

        if not candidate_tokens:
            return 0.0

        smoothie = SmoothingFunction().method5
        bleu_score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=smoothie
        )
        return round(bleu_score * 100, 4)
    except Exception:
        return 0.0

def preprocess_function(examples, tokenizer):
    prompts = []
    for desc, code in zip(examples["description"], examples["code"]):
        prompt = f"<|im_start|>system\nYou are a Solidity expert.<|im_end|>\n<|im_start|>user\n{desc}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        prompts.append(prompt)

    encodings = tokenizer(
        prompts,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )

    labels = encodings["input_ids"].clone()
    response_start_token = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    
    for i in range(len(labels)):
        input_ids = encodings["input_ids"][i].tolist()
        start_idx = None
        for j in range(len(input_ids) - len(response_start_token) + 1):
            if input_ids[j:j+len(response_start_token)] == response_start_token:
                start_idx = j + len(response_start_token)
                break
        if start_idx is not None:
            labels[i, :start_idx] = -100

    labels[encodings["attention_mask"] == 0] = -100

    encodings["labels"] = labels
    return encodings

def load_data(train_path, val_path, tokenizer):
    def load_json(file_path):
        data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    if "description" in item and "code" in item:
                        data.append({
                            "description": str(item["description"]).strip(),
                            "code": str(item["code"]).strip()
                        })
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load dataset: {file_path}") from exc

        if not data:
            raise ValueError(f"Dataset contains no valid description/code pairs: {file_path}")
        return data

    train_data = load_json(train_path)
    val_data = load_json(val_path)

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data)
    })

    tokenized_train = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names
    )
    tokenized_val = dataset["val"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=16,
        remove_columns=dataset["val"].column_names
    )

    return tokenized_train, tokenized_val

class MetricsCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if hasattr(predictions, 'dtype') and predictions.dtype not in [np.int32, np.int64]:
            predictions = np.argmax(predictions, axis=-1)

        predictions = predictions.astype(np.int32)
        labels = labels.astype(np.int32)

        total_bleu = 0.0
        valid_count = 0

        for i in range(predictions.shape[0]):
            sample_pred = predictions[i]
            sample_label = labels[i]

            mask = sample_label != -100
            valid_pred = sample_pred[mask]
            valid_label = sample_label[mask]

            if len(valid_label) == 0:
                continue

            pred_text = self.tokenizer.decode(valid_pred, skip_special_tokens=True).strip()
            label_text = self.tokenizer.decode(valid_label, skip_special_tokens=True).strip()

            if pred_text and label_text:
                bleu_score = calculate_bleu(label_text, pred_text)
                total_bleu += bleu_score
                valid_count += 1

        avg_bleu = total_bleu / valid_count if valid_count > 0 else 0.0
        return {"bleu": avg_bleu}

def main():
    parser = argparse.ArgumentParser(description="qwencoder-sft")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("SFT requires a CUDA-capable NVIDIA GPU for 8-bit model loading.")
    
    set_seed(args.seed)
    logger = setup_logging(args.output_dir)
    
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    train_dataset, val_dataset = load_data(args.train_data_path, args.val_data_path, tokenizer)
    metrics_calculator = MetricsCalculator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=metrics_calculator.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    trainer.train()
    
    final_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

if __name__ == "__main__":
    main()
