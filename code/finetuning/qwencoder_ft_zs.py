import json
import logging
import torch
import os
import time
import argparse
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from peft import PeftModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class QwenCoderLoRAInferencer:
    def __init__(self, args):
        self.base_model_path = args.base_model_path
        self.lora_model_path = args.lora_model_path
        self.results_dir = os.path.abspath(args.results_dir)
        self.log_file = os.path.join(self.results_dir, args.log_file)
        self.results_file = os.path.join(self.results_dir, "qwen_lora_contract_results.json")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("LoRA inference requires a CUDA-capable NVIDIA GPU.")
        self.load_lora_model()
        self.initialize_results_file()
        
    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger

    def initialize_results_file(self):
        initial_data = {
            "model_info": {
                "base_model": self.base_model_path,
                "lora_model": self.lora_model_path,
                "inference_mode": "8bit",
                "quantization_config": "load_in_8bit=True"
            },
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "start_time_timestamp": time.time(),
            "results": []
        }
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

    def append_to_results_file(self, result):
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["results"].append(result)
            data["last_update"] = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error appending result: {str(e)}")

    def update_final_stats(self, total_samples, successful_gens, avg_bleu):
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["end_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
            data["stats"] = {
                "total_samples": total_samples,
                "successful_generations": successful_gens,
                "average_bleu": round(avg_bleu, 4),
                "success_rate": round(successful_gens / total_samples * 100, 2) if total_samples > 0 else 0.0,
                "inference_duration_s": round(time.time() - data.get("start_time_timestamp", time.time()), 2)
            }
            
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error updating stats: {str(e)}")

    def load_lora_model(self):
        try:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left", 
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                use_cache=True
            )
            
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_model_path,
                device_map="auto"
            )
            
            self.model.eval()
            torch.set_grad_enabled(False)
            
            self.generation_config = {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.85,
                "top_k": 50,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.15,
                "num_beams": 1,
                "length_penalty": 1.2,
                "early_stopping": False
            }
            
        except Exception as e:
            raise e

    def create_prompt_template(self, description):
        prompt = f"""### Instruction:
You are an expert Solidity developer.
Please write a high-quality, secure, and complete Solidity smart contract based on the following description.
### Description:
{description}
### Response:
"""
        return prompt

    def generate_contracts_batch(self, descriptions):
        prompts = [self.create_prompt_template(desc) for desc in descriptions]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  
            padding=True,
            return_offsets_mapping=False
        ).to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        full_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        generated_codes = []
        for full_text, prompt in zip(full_texts, prompts):
            response_marker = "### Response:"
            if response_marker in full_text:
                code_part = full_text.split(response_marker)[1].strip()
                generated_codes.append(code_part)
            else:
                generated_codes.append(full_text.strip())
        
        generation_time = round(time.time() - start_time, 2)
        return generated_codes, generation_time

    def calculate_bleu_score(self, reference_code, generated_code):
        try:
            reference_tokens = word_tokenize(reference_code)
            candidate_tokens = word_tokenize(generated_code)
    
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

    def process_test_dataset(self, test_data_path, batch_size):
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except Exception:
            return

        total_samples = len(test_data)
        total_bleu = 0.0
        total_successful = 0
        
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch = test_data[start_idx:end_idx]
            
            descriptions = [sample["description"] for sample in batch]
            sample_indices = [sample["index"] for sample in batch]
            reference_codes = [sample["code"] for sample in batch]
            
            try:
                generated_codes, batch_gen_time = self.generate_contracts_batch(descriptions)
                
                for i in range(len(batch)):
                    sample_idx = sample_indices[i]
                    reference_code = reference_codes[i]
                    generated_code = generated_codes[i]
                    
                    if generated_code:
                        bleu_score = self.calculate_bleu_score(reference_code, generated_code)
                        total_bleu += bleu_score
                        total_successful += 1
                        
                        result = {
                            "index": sample_idx,
                            "filename": batch[i].get("filename", f"sample_{sample_idx}.sol"),
                            "description": descriptions[i],
                            "reference_code": reference_code,
                            "generated_code": generated_code,
                            "bleu_score": bleu_score,
                            "generation_time_s": batch_gen_time / len(batch),
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.append_to_results_file(result)
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
                continue

        avg_bleu = total_bleu / total_successful if total_successful > 0 else 0.0
        self.update_final_stats(total_samples, total_successful, avg_bleu)
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="QwenCoder LoRA Inference")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--lora_model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default="dataset/test/test.json")
    parser.add_argument("--results_dir", type=str, default="./qwen_lora_contract_results")
    parser.add_argument("--log_file", type=str, default="qwen_lora_log.log")
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    inferencer = QwenCoderLoRAInferencer(args)
    inferencer.process_test_dataset(args.test_data_path, args.batch_size)

if __name__ == "__main__":
    main()
