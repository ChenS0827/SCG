import os
import json
import time
import logging
import argparse
import random
import numpy as np
import torch
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def setup_logging(log_file="qwencoder_scot_log.log"):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

        bleu_percentage = bleu_score * 100
        return round(bleu_percentage, 4)

    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

def load_model(model_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

def load_test_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Test data must be a JSON array.")
        
        valid_samples = []
        for sample in data:
            if all(k in sample for k in ["index", "description"]):
                if "code" not in sample and "reference_code" in sample:
                    sample["code"] = sample["reference_code"]
                
                if "code" in sample:
                    valid_samples.append(sample)
        
        return valid_samples
    except Exception as e:
        raise

def build_scot_prompt(description):
    system_prompt = """### Instruction: You are a Solidity expert. Analyze the requirements step-by-step using Structured Chain-of-Thought before generating code."""
    
    examples_section = """
### Example
### Requirement: Create a simple storage contract to store a number.
### SCOT Reasoning:
1. Interface Analysis
Input: uint256 num
Output: None
2. Control Flow Planning
Sequential:
Step 1: Create a state variable to store the number.
Step 2: Create a function to set the number.
Step 3: Create a function to get the number.
Branch (If/Else): None
Loop (For/While): None
3. Security & Finalization
Security Checks: None
Modifiers: None
Finalize: Logic is complete.
### Response:
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 public storedNumber;

    function setNumber(uint256 _num) public {
        storedNumber = _num;
    }

    function getNumber() public view returns (uint256) {
        return storedNumber;
    }
}
"""
    current_task = f"""
Requirement: {description}
"""
    return f"{system_prompt}\n{examples_section}\n{current_task}"

def scot_generate(tokenizer, model, scot_prompt, max_new_tokens=4096):
    try:
        if isinstance(scot_prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(
                scot_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = scot_prompt

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 
        ).to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - start_time
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if full_output.startswith(prompt_text):
             model_output = full_output[len(prompt_text):].strip()
        else:
             model_output = full_output

        return model_output, round(gen_time, 2)
        
    except Exception:
        return "", 0.0

def extract_content(model_output):
    scot_reasoning = ""
    generated_code = ""

    reasoning_patterns = [
        r'### SCOT Reasoning:(.*?)### Response:',
        r'SCoT Reasoning:(.*?)Generated Code:'
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, model_output, re.DOTALL | re.IGNORECASE)
        if match:
            scot_reasoning = match.group(1).strip()
            break
            
    scot_reasoning = scot_reasoning.replace("**", "").strip()
    
    code_match = re.search(r'```solidity\s*(.*?)```', model_output, re.DOTALL | re.IGNORECASE)
    if code_match:
        generated_code = code_match.group(1).strip()
    else:
        start_match = re.search(r'```solidity\s*', model_output, re.IGNORECASE)
        if start_match:
            generated_code = model_output[start_match.end():].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3].strip()

    return scot_reasoning, generated_code

def update_results(output_path, new_sample, is_first=False):
    logger = logging.getLogger(__name__)
    
    if os.path.exists(output_path) and not is_first:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            samples = results.get("results", [])
            samples.append(new_sample)
        except json.JSONDecodeError:
            samples = [new_sample]
            results = {} 
    else:
        samples = [new_sample]
        results = {}

    success_samples = [s for s in samples if s.get("generated_code")]
    total = len(samples)
    success_count = len(success_samples)
    avg_bleu = 0.0
    if success_count > 0:
        avg_bleu = sum(s["bleu_score"] for s in success_samples) / success_count

    results["summary"] = {
        "total_samples": total,
        "successful_generations": success_count,
        "success_rate_percent": round(success_count / total * 100, 2) if total > 0 else 0,
        "average_bleu": round(avg_bleu, 4),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    results["results"] = samples
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results updated in {output_path} (Total: {total})")

def main():
    parser = argparse.ArgumentParser(description="qwencoder-scot")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Path or Hugging Face model ID")
    parser.add_argument("--data_path", type=str, default="dataset/test/test.json", help="Path to the test dataset JSON file")
    parser.add_argument("--output_path", type=str, default="qwencoder_scot_results/qwencoder_scot_result.json", help="Path to save the evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_file", type=str, default="qwencoder_scot_log.log", help="Path to the log file")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    logger = setup_logging(args.log_file)
    set_seed(args.seed)
    
    try:
        model, tokenizer = load_model(args.model_path)
        test_samples = load_test_data(args.data_path)
        
        logger.info(f"Starting SCoT evaluation on {len(test_samples)} samples.")

        for i, sample in enumerate(tqdm(test_samples), 1):
            idx = sample.get("index", i)
            filename = sample.get("filename", "unknown")
            description = sample["description"]
            reference_code = sample["code"]

            scot_prompt = build_scot_prompt(description)
            
            generated_output, gen_time = scot_generate(tokenizer, model, scot_prompt)
            
            scot_reasoning, generated_code = extract_content(generated_output)
            
            bleu_score = calculate_bleu(reference_code, generated_code)

            sample_result = {
                "index": idx,
                "filename": filename,
                "description": description,
                "reference_code": reference_code,
                "generated_code": generated_code,
                "scot_reasoning": scot_reasoning,
                "bleu_score": bleu_score,
                "generate_time_seconds": gen_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            update_results(args.output_path, sample_result, is_first=(i == 1))
            
            logger.info(f"Sample {idx} - BLEU Score: {bleu_score}%")

        logger.info("Evaluation finished.")

    except Exception as e:
        logger.critical(f"Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
