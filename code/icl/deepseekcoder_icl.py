import os
import json
import logging
import argparse
import random
import re
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def setup_logger(log_file):
    logger = logging.getLogger('deepseek_icl_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score and convert to percentage format.
    """
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

def load_json_data(path):
    """
    Load JSON data (ICL examples or test data).
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise e

def extract_solidity_code(text):
    """
    Extract Solidity code from the generated text.
    """
    if not text:
        return ""


    pattern = r'```solidity(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[-1].strip()

    pattern_generic = r'```(.*?)```'
    matches_generic = re.findall(pattern_generic, text, re.DOTALL)
    
    if matches_generic:
        code = matches_generic[-1].strip()
        if code.lower().startswith('solidity'):
            return code[8:].strip()
        return code

    return text.strip()

def build_icl_prompt(icl_examples, description):
    """
    Build prompt based on the provided PDF template (Figure 7).
    """

    prompt = "### Instruction:\n"
    prompt += "You are an expert Solidity developer.\n"
    prompt += "Please write a high-quality, secure, and complete Solidity smart contract based on the following description.\n"
    
    prompt += "### Examples:\n"
    for example in icl_examples:
        prompt += f"Description: {example.get('description', '')}\n"
        prompt += f"Code: {example.get('code', '')}\n\n"
    
    prompt += "### Description:\n"
    prompt += f"{description}\n"
    
    prompt += "### Response:\n"
    
    return prompt

def generate_code_batch(model, tokenizer, prompts, batch_size=1, max_length=8000, temperature=0.2, top_p=0.95, logger=None):
    """
    Generate code in batches.
    """
    generated_codes = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        if logger:
            logger.info(f"Generating batch {i//batch_size + 1}, size {len(batch_prompts)}")
        
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                pad_to_multiple_of=8
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1800,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            for j, output in enumerate(outputs):
                generated_full = tokenizer.decode(output, skip_special_tokens=True)
                
                response_marker = "### Response:"
                if response_marker in generated_full:
                    model_output = generated_full.split(response_marker)[-1].strip()
                else:
                    
                    prompt_len = len(tokenizer.decode(inputs["input_ids"][j], skip_special_tokens=True))
                    model_output = generated_full[prompt_len:].strip()
                
                extracted_code = extract_solidity_code(model_output)
                generated_codes.append(extracted_code)
                
        except Exception as e:
            if logger:
                logger.error(f"Error in batch generation: {e}")
            for _ in batch_prompts:
                generated_codes.append("")
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return generated_codes

def main():
    parser = argparse.ArgumentParser(description="deepseek-coder-zero-shot-icl")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset json")
    parser.add_argument("--icl_examples_path", type=str, required=True, help="Path to the ICL examples json")
    parser.add_argument("--output_dir", type=str, default="deepseek_contract_results", help="Directory to save results")
    parser.add_argument("--load_in_8bit", "--do_train", dest="load_in_8bit", action="store_true", help="Load the model in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "deepseek_icl_log.log")
    logger = setup_logger(log_file)
    
    set_seed(args.seed)
    logger.info(f"Arguments: {args}")


    try:
        icl_examples = load_json_data(args.icl_examples_path)
        test_data = load_json_data(args.test_data_path)
        logger.info(f"Loaded {len(icl_examples)} ICL examples and {len(test_data)} test samples.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return


    logger.info("Loading model...")
    if args.load_in_8bit:

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()

    results = []
    prompts = []
    samples_to_process = []
    
    for sample in test_data:
        description = sample.get('description', '')
        prompt = build_icl_prompt(icl_examples, description)
        prompts.append(prompt)
        samples_to_process.append(sample)

    logger.info(f"Starting generation for {len(prompts)} samples...")
    generated_codes = generate_code_batch(
        model, 
        tokenizer, 
        prompts, 
        batch_size=args.batch_size,
        logger=logger
    )

    total_bleu = 0.0
    
    for i, (sample, gen_code) in enumerate(zip(samples_to_process, generated_codes)):
        reference_code = sample.get('code', '') # Assuming 'code' is the key in test data
        
        bleu = calculate_bleu(reference_code, gen_code)
        total_bleu += bleu
        
        result_entry = {
            "index": sample.get('index', i),
            "description": sample.get('description', ''),
            "reference_code": reference_code,
            "generated_code": gen_code,
            "bleu_score": bleu
        }
        results.append(result_entry)
        
        if i % 10 == 0:
            logger.info(f"Sample {i}: BLEU={bleu:.4f}")

    avg_bleu = total_bleu / len(results) if results else 0
    
    final_data = {
        "summary": {
            "average_bleu_score": avg_bleu,
            "total_samples": len(results),
            "args": vars(args),
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    output_file = os.path.join(args.output_dir, "deepseek_icl_result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Finished. Average BLEU: {avg_bleu:.4f}. Results saved to {output_file}")

if __name__ == "__main__":
    main()
