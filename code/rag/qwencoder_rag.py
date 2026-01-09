import os
import json
import logging
import re
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def setup_logger(log_file):
    logger = logging.getLogger('qwencoder_rag_logger')
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

def load_retrieval_results(path):
    """
    Load retrieval results from JSON.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        if isinstance(retrieval_data, list):
            test_samples = retrieval_data
        else:
            test_samples = retrieval_data
            if 'results' in retrieval_data:
                test_samples = retrieval_data['results']
        
        return test_samples
    except Exception as e:
        raise e

def count_tokens(text, tokenizer):
    """Calculate token count."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_code_by_tokens(code, tokenizer, max_tokens=1900):
    """
    Truncate code to specified token count, preserving structure where possible.
    """
    if not code:
        return code
    
    current_tokens = count_tokens(code, tokenizer)
    if current_tokens <= max_tokens:
        return code
    
    lines = code.split('\n')
    truncated_lines = []
    current_token_count = 0

    important_keywords = ['pragma', 'import', 'contract', 'function', 'event', 'modifier', 'struct', 'enum']
    

    for line in lines:
        if any(keyword in line for keyword in important_keywords):
            line_tokens = count_tokens(line, tokenizer)
            if current_token_count + line_tokens <= max_tokens:
                truncated_lines.append(line)
                current_token_count += line_tokens
            else:
                break
    

    if current_token_count < max_tokens:
        for line in lines:
            if line not in truncated_lines:
                line_tokens = count_tokens(line, tokenizer)
                if current_token_count + line_tokens <= max_tokens:
                    truncated_lines.append(line)
                    current_token_count += line_tokens
                else:

                    remaining_tokens = max_tokens - current_token_count
                    if remaining_tokens > 10:
                        words = line.split()
                        partial_line = []
                        for word in words:
                            word_tokens = count_tokens(word, tokenizer)
                            if current_token_count + word_tokens <= max_tokens:
                                partial_line.append(word)
                                current_token_count += word_tokens
                            else:
                                break
                        if partial_line:
                            truncated_lines.append(' '.join(partial_line))
                    break
    
    return '\n'.join(truncated_lines)

def process_retrieved_contexts(retrieved_contexts, top_k, tokenizer):
    """
    Process retrieved contexts: truncate if >= 2000 tokens.
    """
    if not retrieved_contexts:
        return []
    
    processed_contexts = []
    
    for i, ctx in enumerate(retrieved_contexts[:top_k]):

        example_text = f"Example Description: {ctx.get('title', '')}\nExample Code:\n```solidity\n{ctx['text']}\n```"
        token_count = count_tokens(example_text, tokenizer)
        
        if token_count >= 2000:
            truncated_code = truncate_code_by_tokens(ctx['text'], tokenizer, 1900)
            
            truncated_example_text = f"Example Description: {ctx.get('title', '')}\nExample Code:\n```solidity\n{truncated_code}\n```"
            truncated_token_count = count_tokens(truncated_example_text, tokenizer)
            
            processed_contexts.append({
                'title': ctx.get('title', ''),
                'text': truncated_code,
                'original_tokens': token_count,
                'truncated_tokens': truncated_token_count,
                'truncated': True
            })
        else:
            processed_contexts.append({
                'title': ctx.get('title', ''),
                'text': ctx['text'],
                'original_tokens': token_count,
                'truncated_tokens': token_count,
                'truncated': False
            })
    
    return processed_contexts

def remove_comments(code):
    """
    Remove Solidity comments (// and /* */).
    """
    if not code:
        return ""
    
    try:

        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        code = re.sub(r'//.*', '', code)

        code = re.sub(r'\n\s*\n', '\n\n', code)
        return code.strip()
    except Exception:
        return code

def extract_solidity_code(generated_text, sample_index, logger=None):
    """
    Extract Solidity code from markdown blocks or raw text.
    """
    if not generated_text:
        return ""

    pattern = r'```solidity(.*?)```'
    matches = re.findall(pattern, generated_text, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        return remove_comments(code)
    

    pattern_fallback = r'```(.*?)```'
    matches_fallback = re.findall(pattern_fallback, generated_text, re.DOTALL)
    
    if matches_fallback:
        code = matches_fallback[0].strip()
        if code.lower().startswith('solidity'):
            code = code[8:].strip()
        return remove_comments(code)
    

    if logger:
        logger.info(f"Sample {sample_index}: Using full text fallback.")
    return remove_comments(generated_text.strip())

def build_prompt_with_rag(retrieved_contexts, question, tokenizer, top_k=None):
    """
    Build prompt using the standardized Figure 7 template + RAG Contexts.
    Returns a raw string.
    """
    # Process contexts
    if top_k is not None and retrieved_contexts:
        used_contexts = process_retrieved_contexts(retrieved_contexts, top_k, tokenizer)
    else:
        used_contexts = retrieved_contexts
    

    prompt = "### Instruction:\n"
    prompt += "You are an expert Solidity developer.\n"
    prompt += "Please write a high-quality, secure, and complete Solidity smart contract based on the following description.\n\n"


    if used_contexts:
        prompt += "Reference Examples:\n"
        for idx, ctx in enumerate(used_contexts, 1):
            truncation_note = " [TRUNCATED]" if ctx.get('truncated', False) else ""
            prompt += f"Example {idx}: {ctx['title']}{truncation_note}\n"
            prompt += f"```solidity\n{ctx['text']}\n```\n\n"
    

    prompt += "### Description:\n"
    prompt += f"{question}\n\n"
    

    prompt += "### Response:\n"
    
    return prompt

def generate_code_batch(model, tokenizer, prompts, batch_size=4, max_length=10000, temperature=0.3, top_p=0.9, logger=None):
    """
    Generate code in batches using raw string prompts.
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
                    max_new_tokens=2048,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2,
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

                sample_index = i + j + 1
                extracted_code = extract_solidity_code(model_output, sample_index, logger)
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
    parser = argparse.ArgumentParser(description="qwencoder-rag")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to retrieval results json")
    parser.add_argument("--output_dir", type=str, default="qwencoder_rag_results", help="Directory to save results")
    parser.add_argument("--do_train", action="store_true", help="Enable training mode (uses 8-bit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--top_k", type=int, default=4, help="Number of retrieval contexts to use")
    
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "generation.log")
    logger = setup_logger(log_file)
    
    set_seed(args.seed)
    logger.info(f"Arguments: {args}")


    logger.info("Loading model...")
    if args.do_train:
        
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Inference: Full parameters
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    # 3. Load Data
    try:
        test_samples = load_retrieval_results(args.data_path)
        logger.info(f"Loaded {len(test_samples)} samples.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 4. Processing Loop
    results = []
    prompts = []
    samples_to_process = []
    
    # Prepare prompts
    for sample in test_samples:
        retrieved_contexts = sample.get("ctxs", [])
        question = sample.get("question", "")
        
        prompt = build_prompt_with_rag(retrieved_contexts, question, tokenizer, top_k=args.top_k)
        prompts.append(prompt)
        samples_to_process.append(sample)

    # Generate
    logger.info("Starting generation...")
    generated_codes = generate_code_batch(
        model, 
        tokenizer, 
        prompts, 
        batch_size=args.batch_size,
        logger=logger
    )

    # 5. Calculate Metrics and Save
    total_bleu = 0.0
    successful_generations = 0
    
    for i, (sample, gen_code) in enumerate(zip(samples_to_process, generated_codes)):
        reference_code = sample.get("target", "")
        
        bleu = calculate_bleu(reference_code, gen_code)
        total_bleu += bleu
        
        is_success = bool(gen_code and len(gen_code) > 50)
        if is_success:
            successful_generations += 1
            
        result_entry = {
            "index": sample.get("index", i),
            "description": sample.get("question", ""),
            "reference_code": reference_code,
            "generated_code": gen_code,
            "bleu_score": bleu,
            "generation_success": is_success
        }
        results.append(result_entry)
        
        logger.info(f"Sample {i}: BLEU={bleu:.4f}, Success={is_success}")

    # Final Statistics
    avg_bleu = total_bleu / len(results) if results else 0
    success_rate = (successful_generations / len(results) * 100) if results else 0
    
    summary = {
        "average_bleu": avg_bleu,
        "success_rate": success_rate,
        "total_samples": len(results),
        "args": vars(args),
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Finished. Avg BLEU: {avg_bleu:.4f}. Results saved to {output_file}")

if __name__ == "__main__":
    main()