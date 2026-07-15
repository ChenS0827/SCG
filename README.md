# Repository-Level Solidity Code Generation with Large Language Models: From Prompting to Fine-Tuning

This repository contains the dataset and experimental code accompanying the paper **“Repository-Level Solidity Code Generation with Large Language Models: From Prompting to Fine-Tuning.”** The study examines how three open-source code language models generate complete Solidity contracts from natural-language specifications and compares five adaptation settings: zero-shot prompting, Structured Chain-of-Thought (SCoT), in-context learning (ICL), retrieval-augmented generation (RAG), and supervised fine-tuning (SFT).

The repository also releases **SolidityBench**, a benchmark of 5,470 repository-level Solidity samples paired with natural-language descriptions. The benchmark was curated from OpenZeppelin, Synthetix, and verified contracts on Etherscan, and is intended to preserve the contract structure, inheritance relationships, and cross-contract dependencies found in real-world Solidity projects.

## What is included

```text
SCG/
├── code/
│   ├── zeroshot/       # Zero-shot generation
│   ├── cot/            # Structured Chain-of-Thought prompting
│   ├── icl/            # In-context learning
│   ├── rag/            # Generation from precomputed retrieval results
│   └── finetuning/     # LoRA fine-tuning and inference
├── dataset/
│   ├── train/train.json
│   ├── val/val.json
│   └── test/test.json
└── README.md
```

Each dataset record contains four fields:

| Field | Description |
| --- | --- |
| `index` | Record identifier within the split |
| `filename` | Original Solidity filename |
| `description` | Natural-language specification of the contract |
| `code` | Reference Solidity implementation |

The released split follows the 8:1:1 partition used in the paper:

| Split | Samples | Proportion |
| --- | ---: | ---: |
| Training | 4,376 | 80% |
| Validation | 547 | 10% |
| Test | 547 | 10% |
| **Total** | **5,470** | **100%** |

## Models and experimental settings

The experiments evaluate three instruction-tuned models of comparable scale:

- `meta-llama/CodeLlama-7b-Instruct-hf`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `Qwen/Qwen2.5-Coder-7B-Instruct`

Five settings are represented in the code:

1. **Zero-shot:** the model receives only the target contract description.
2. **SCoT:** the prompt asks the model to analyze interfaces, control flow, and security constraints before producing code.
3. **ICL:** one to four randomly selected training examples are supplied as demonstrations.
4. **RAG:** BM25 retrieves training examples whose descriptions are similar to the target query, and the retrieved examples are added to the prompt.
5. **SFT:** LoRA adapters are trained on the Solidity instruction data and evaluated on the held-out test set.

The paper reports results with Python 3.12 and PyTorch 2.8.0 on a single NVIDIA GeForce RTX 5090 with 32 GB of VRAM. Inference uses a temperature of 0.2. The SFT configuration uses LoRA with rank 16 and scaling factor 32, a learning rate of `3e-4`, cosine scheduling, a warmup ratio of `0.05`, and loss masking over the instruction prefix.

## Environment

The scripts depend on the following Python packages:

```text
torch
transformers
accelerate
bitsandbytes
peft
datasets
numpy
nltk
tqdm
```

Create an isolated environment before installing the dependencies. CUDA-capable NVIDIA hardware is required by the current quantized loading and training configuration.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model weights are not stored in this repository. Download them from their official Hugging Face releases or pass the path of an existing local copy through the relevant command-line option.

## Running the experiments

Run commands from the repository root. The examples below use Qwen2.5-Coder; the corresponding CodeLlama and DeepSeek-Coder scripts accept similar arguments.

### Zero-shot generation

```bash
python code/zeroshot/qwencoder_zs.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --data_path dataset/test/test.json \
  --output_path outputs/qwen_zero_shot/results.json \
  --log_file outputs/qwen_zero_shot/run.log
```

### In-context learning

The ICL scripts expect a test set and a separate JSON file containing demonstration examples. Use records from the training split for demonstrations; do not sample demonstrations from the validation or test split.

```bash
python code/icl/qwencoder_icl.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --test_data_path dataset/test/test.json \
  --icl_examples_path path/to/icl_examples.json \
  --output_dir outputs/qwen_icl \
  --batch_size 1
```

### Retrieval-augmented generation

The RAG scripts consume precomputed retrieval results rather than building the BM25 index themselves. Each test record in the input JSON must provide the target specification in `question`, the reference implementation in `target`, and a `ctxs` list containing the retrieved examples. Each context must contain `title` (the example specification) and `text` (the Solidity implementation).

```bash
python code/rag/qwencoder_rag.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --data_path path/to/retrieval_results.json \
  --output_dir outputs/qwen_rag \
  --top_k 2 \
  --batch_size 1
```

### Supervised fine-tuning

```bash
python code/finetuning/qwencoder_ft.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --train_data_path dataset/train/train.json \
  --val_data_path dataset/val/val.json \
  --output_dir outputs/qwen_sft \
  --num_train_epochs 30 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 5 \
  --learning_rate 3e-4 \
  --warmup_ratio 0.05 \
  --early_stopping_patience 2 \
  --seed 42
```

After training, use `code/finetuning/qwencoder_ft_zs.py` to evaluate the saved LoRA adapter on the test set.

## Evaluation

The paper reports two principal metrics:

- **BLEU**, which measures lexical overlap between generated and reference code.
- **SolidityScore**, a semantics-aware metric designed for Solidity. It combines representations from a Solidity-adapted encoder with domain-weighted token matching so that Solidity-specific constructs receive appropriate emphasis.

The paper additionally studies compilation success and manually analyzes Solidity-specific structural, logical, and security errors. The current generation scripts calculate BLEU during inference. The implementation of SolidityScore and the compilation-analysis pipeline are not included in this repository at present.

## Main findings

Across the evaluated models, zero-shot generation frequently produces structural, syntactic, and security-related defects. SCoT and ICL improve generation quality, although ICL performance generally declines when the prompt contains more than two demonstrations. RAG performs best among the inference-time adaptation methods because it supplies examples that are relevant to each query, but it also exhibits context saturation as more examples are added. SFT provides the strongest overall results by incorporating Solidity-specific patterns directly into the model parameters.

These results should not be interpreted as evidence that generated contracts are ready for deployment. Smart contracts produced by language models still require compilation, testing, security analysis, and expert review.

## Reproducibility notes

- Use the released train, validation, and test partitions without moving examples between splits.
- Fix the random seed when comparing different models or prompting strategies.
- Keep decoding parameters consistent across methods.
- Record the exact model revision, package versions, GPU type, and retrieval configuration used for each run.
- The scripts write generated code and BLEU scores to JSON files; preserve these files when conducting later SolidityScore or compilation analysis.

## Citation

If you use SolidityBench or the accompanying experimental code, please cite the paper:

```bibtex
@article{chen_repository_level_solidity_generation,
  title   = {Repository-Level Solidity Code Generation with Large Language Models: From Prompting to Fine-Tuning},
  author  = {Chen, Shi and Wang, Rongcun and Tian, Yuan and Xie, Xiaoyuan and Song, Wei and Huang, Rubing},
  note    = {Manuscript}
}
```

## License

No license file is currently included. Unless a license is added, the repository should not be assumed to grant permission for reuse, modification, or redistribution beyond what is permitted by applicable law.
