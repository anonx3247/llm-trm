#!/usr/bin/env python3
"""
Math Benchmark Script for LLM-TRM

Evaluates SmolLM3 on GSM8K or AIME with different configurations:
- base (--thinking off): SmolLM3 without thinking mode
- base (--thinking on): SmolLM3 with native chain-of-thought
- compressor: SmolLM3 with hidden states passed through compressor
- trm: SmolLM3 with TRM replacing thinking

Usage:
    python scripts/benchmark.py --model base --thinking on --dataset gsm8k --num_samples 100
    python scripts/benchmark.py --model base --thinking on --dataset aime --num_samples 30
    python scripts/benchmark.py --model trm --dataset aime --num_samples 30
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.aime import (
    extract_answer as extract_answer_aime,
    format_aime_prompt,
    is_correct as is_correct_aime,
    load_aime,
)
from src.evaluation.gsm8k import (
    extract_answer as extract_answer_gsm8k,
    format_gsm8k_prompt,
    is_correct as is_correct_gsm8k,
    load_gsm8k,
)
from src.models.inference import CompressorOnlyInference, SmolLMWithTRMInference


# Dataset utilities - selected based on --dataset arg
def get_dataset_utils(dataset: str) -> tuple[
    Callable[[str], list[dict[str, Any]]],  # load_fn
    Callable[[str], str],  # format_prompt_fn
    Callable[[str], str | None],  # extract_answer_fn
    Callable[[str | None, str | None], bool],  # is_correct_fn
]:
    """Get dataset-specific utility functions."""
    if dataset == "gsm8k":
        return (
            lambda n: load_gsm8k(split="test", num_samples=n),
            format_gsm8k_prompt,
            extract_answer_gsm8k,
            is_correct_gsm8k,
        )
    elif dataset == "aime":
        return (
            lambda n: load_aime(num_samples=n),
            format_aime_prompt,
            extract_answer_aime,
            is_correct_aime,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def evaluate_base_model(
    samples: list[dict[str, Any]],
    thinking: bool,
    format_prompt: Callable[[str], str],
    extract_answer: Callable[[str], str | None],
    is_correct: Callable[[str | None, str | None], bool],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate base SmolLM3 model."""
    model_name = "HuggingFaceTB/SmolLM3-3B"

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model = model.to(device)
    else:
        device = "cpu"
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    results = []
    correct_count = 0
    pbar = tqdm(samples, desc=f"Base (think={thinking}) acc=0.0%")
    for sample in pbar:
        question = sample["question"]
        gold = sample["gold"]

        # Format prompt
        prompt = format_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

        # Generate
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        # Extract answer and check correctness
        predicted = extract_answer(response)
        correct = is_correct(predicted, gold)
        if correct:
            correct_count += 1
        acc = correct_count / (len(results) + 1) * 100
        pbar.set_description(f"Base (think={thinking}) acc={acc:.1f}%")

        results.append(
            {
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
                "response": response,
                "tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
                "time": gen_time,
            }
        )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def evaluate_compressor_model(
    samples: list[dict[str, Any]],
    format_prompt: Callable[[str], str],
    extract_answer: Callable[[str], str | None],
    is_correct: Callable[[str | None, str | None], bool],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate SmolLM3 with compressor roundtrip."""
    model = CompressorOnlyInference()

    results = []
    correct_count = 0
    pbar = tqdm(samples, desc="Compressor acc=0.0%")
    for sample in pbar:
        question = sample["question"]
        gold = sample["gold"]
        prompt = format_prompt(question)

        start_time = time.time()
        result = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            enable_thinking=True,
        )
        gen_time = time.time() - start_time

        # Extract assistant response
        response = result["text"]
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        # Extract answer and check correctness
        predicted = extract_answer(response)
        correct = is_correct(predicted, gold)
        if correct:
            correct_count += 1
        acc = correct_count / (len(results) + 1) * 100
        pbar.set_description(f"Compressor acc={acc:.1f}%")

        results.append(
            {
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
                "response": response,
                "tokens": result["tokens_generated"],
                "time": gen_time,
            }
        )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def evaluate_trm_model(
    samples: list[dict[str, Any]],
    format_prompt: Callable[[str], str],
    extract_answer: Callable[[str], str | None],
    is_correct: Callable[[str | None, str | None], bool],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate SmolLM3 with TRM (/no_think + TRM enhancement)."""
    model = SmolLMWithTRMInference()

    results = []
    correct_count = 0
    pbar = tqdm(samples, desc="TRM acc=0.0%")
    for sample in pbar:
        question = sample["question"]
        gold = sample["gold"]
        prompt = format_prompt(question)

        start_time = time.time()
        result = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        gen_time = time.time() - start_time

        # Extract assistant response
        response = result["text"]
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        # Extract answer and check correctness
        predicted = extract_answer(response)
        correct = is_correct(predicted, gold)
        if correct:
            correct_count += 1
        acc = correct_count / (len(results) + 1) * 100
        pbar.set_description(f"TRM acc={acc:.1f}%")

        results.append(
            {
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
                "response": response,
                "tokens": result["tokens_generated"],
                "time": gen_time,
            }
        )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Math Benchmark for LLM-TRM")
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "compressor", "trm"],
        required=True,
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "aime"],
        default="gsm8k",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable thinking mode (only for base model)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate (default: None = no limit, uses 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Handle max_tokens default
    if args.max_tokens is None:
        args.max_tokens = 8192  # Effectively no limit

    # Validate arguments
    if args.model != "base" and args.thinking == "off":
        print("Warning: --thinking is only used with --model base")

    # Get dataset utilities
    load_fn, format_prompt, extract_answer, is_correct = get_dataset_utils(args.dataset)

    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    samples = load_fn(args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Run evaluation
    print(f"\nEvaluating {args.model} model on {args.dataset.upper()}...")
    if args.model == "base":
        thinking = args.thinking == "on"
        results = evaluate_base_model(
            samples,
            thinking=thinking,
            format_prompt=format_prompt,
            extract_answer=extract_answer,
            is_correct=is_correct,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_name = f"base_thinking_{args.thinking}"
    elif args.model == "compressor":
        results = evaluate_compressor_model(
            samples,
            format_prompt=format_prompt,
            extract_answer=extract_answer,
            is_correct=is_correct,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_name = "compressor"
    elif args.model == "trm":
        results = evaluate_trm_model(
            samples,
            format_prompt=format_prompt,
            extract_answer=extract_answer,
            is_correct=is_correct,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_name = "trm"
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    # Calculate metrics
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    avg_tokens = sum(r["tokens"] for r in results) / len(results) if results else 0
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Avg time: {avg_time:.2f}s")

    # Prepare output
    output_data = {
        "model": model_name,
        "dataset": args.dataset,
        "thinking": args.thinking == "on" if args.model == "base" else None,
        "num_samples": len(results),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "avg_tokens": avg_tokens,
        "avg_time_per_sample": avg_time,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.dataset}_{model_name}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 3)")
    print("=" * 60)
    for i, r in enumerate(results[:3]):
        print(f"\n[{i+1}] Question: {r['question'][:80]}...")
        print(f"    Gold: {r['gold']}")
        print(f"    Predicted: {r['predicted']}")
        print(f"    Correct: {'✓' if r['correct'] else '✗'}")


if __name__ == "__main__":
    main()
