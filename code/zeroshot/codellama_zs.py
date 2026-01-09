import os
import json
import time
import logging
import argparse
import random
import numpy as np
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def setup_logging(log_file="zero_shot_bleu_evaluation.log"):
    """
    Configure logging to output to both console and file.
    """
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
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score and convert to percentage format.
    Uses nltk.word_tokenize and SmoothingFunction method5.
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


def load_model(model_path):
    """
    Load the model and tokenizer.
    For non-training (inference), load with full parameters (no quantization).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Set pad_token to eos_token as open-ended generation models often lack a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        logger.info("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise


def load_test_data(data_path):
    """
    Load test dataset from JSON file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading test data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("test.json must be a JSON array.")
        
        valid_samples = []
        for sample in data:
            if all(k in sample for k in ["index", "filename", "description", "code"]):
                valid_samples.append(sample)
            else:
                logger.warning(f"Skipping invalid sample index: {sample.get('index', 'unknown')}")
        
        logger.info(f"Loaded {len(valid_samples)} valid samples.")
        return valid_samples
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}", exc_info=True)
        raise


def generate_code(tokenizer, model, description, max_new_tokens=512):
    """
    Generate code using the zero-shot prompt template.
    Template based on fig7-prompt_template.pdf.
    """
    logger = logging.getLogger(__name__)
    
    
    prompt = f"""### Instruction:
You are an expert Solidity developer.
Please write a high-quality, secure, and complete Solidity smart contract based on the following description.
### Description:
{description}
### Response:
"""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,       
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen_time = round(time.time() - start_time, 2)
        
        # Decode response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the part after "### Response:"
        response_marker = "### Response:"
        if response_marker in full_output:
            generated_code = full_output.split(response_marker)[1].strip()
        else:
            generated_code = full_output.strip()

        return generated_code, gen_time
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return "", 0.0


def update_results(output_path, new_sample, is_first=False):
    """
    Update the result JSON file incrementally.
    """
    logger = logging.getLogger(__name__)
    
    if os.path.exists(output_path) and not is_first:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            samples = results.get("sample_results", [])
            samples.append(new_sample)
        except json.JSONDecodeError:
            samples = [new_sample]
            results = {} 
    else:
        samples = [new_sample]
        results = {}

    # Update statistics
    success_samples = [s for s in samples if s["status"] == "success"]
    total = len(samples)
    success_count = len(success_samples)
    avg_bleu = 0.0
    if success_count > 0:
        avg_bleu = sum(s["bleu_score"] for s in success_samples) / success_count

    results["overall_statistics"] = {
        "total_samples": total,
        "success_samples": success_count,
        "success_rate_percent": round(success_count / total * 100, 2) if total > 0 else 0,
        "average_bleu": round(avg_bleu, 4),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    results["sample_results"] = samples
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results updated in {output_path} (Total: {total})")


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="codellama-zero-shot")
    parser.add_argument("--model_path", type=str, default="meta-llama/CodeLlama-7B-hf", 
                        help="Path to the pre-trained model")
    parser.add_argument("--data_path", type=str, default="test.json", 
                        help="Path to the test dataset json file")
    parser.add_argument("--output_path", type=str, default="results/zero_shot_results.json", 
                        help="Path to save the evaluation results")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--log_file", type=str, default="zero_shot_evaluation.log", 
                        help="Path to the log file")
    
    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_file)
    set_seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        # Load resources
        tokenizer, model = load_model(args.model_path)
        test_samples = load_test_data(args.data_path)
        
        total_samples = len(test_samples)
        logger.info(f"Starting zero-shot evaluation on {total_samples} samples.")

        for i, sample in enumerate(test_samples, 1):
            idx = sample["index"]
            filename = sample.get("filename", "unknown")
            description = sample["description"]
            reference_code = sample["code"]

            logger.info(f"Processing sample {i}/{total_samples} (Index: {idx})")

            # Generate Code
            generated_code, gen_time = generate_code(tokenizer, model, description)

            # Calculate BLEU using the specific requested function
            bleu_score = calculate_bleu(reference_code, generated_code)

            # Result object
            sample_result = {
                "index": idx,
                "filename": filename,
                "description": description,
                "reference_code": reference_code,
                "generated_code": generated_code,
                "bleu_score": bleu_score,
                "generate_time_seconds": gen_time,
                "status": "success" if generated_code else "failed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save result
            update_results(args.output_path, sample_result, is_first=(i == 1))
            
            logger.info(f"BLEU Score: {bleu_score}%")

        logger.info("Evaluation finished.")

    except Exception as e:
        logger.critical(f"Process failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()