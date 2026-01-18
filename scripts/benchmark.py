#!/usr/bin/env python3
"""
GSM8K Benchmark Script for LLM-TRM

Evaluates SmolLM3 on GSM8K with different configurations:
- base (--thinking off): SmolLM3 without thinking mode
- base (--thinking on): SmolLM3 with native chain-of-thought
- compressor: SmolLM3 with hidden states passed through compressor
- trm: SmolLM3 with TRM replacing thinking

Usage:
    python scripts/benchmark.py --model base --thinking off --num_samples 100
    python scripts/benchmark.py --model base --thinking on --num_samples 100
    python scripts/benchmark.py --model compressor --num_samples 100
    python scripts/benchmark.py --model trm --num_samples 100
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
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.gsm8k import (
    extract_answer,
    format_gsm8k_prompt,
    is_correct,
    load_gsm8k,
)
from src.models.inference import CompressorOnlyInference, SmolLMWithTRMInference


def evaluate_base_model(
    samples: list[dict[str, Any]],
    thinking: bool,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate base SmolLM3 model on GSM8K."""
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
        prompt = format_gsm8k_prompt(question)
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
        prompt = format_gsm8k_prompt(question)

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
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate SmolLM3 with TRM replacing thinking."""
    model = SmolLMWithTRMInference()

    results = []
    correct_count = 0
    trm_activations = 0
    pbar = tqdm(samples, desc="TRM acc=0.0%")
    for sample in pbar:
        question = sample["question"]
        gold = sample["gold"]
        prompt = format_gsm8k_prompt(question)

        start_time = time.time()
        result = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            enable_thinking=True,
        )
        gen_time = time.time() - start_time

        if result["trm_activated"]:
            trm_activations += 1

        # Reset for next sample
        model.reset()

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
                "trm_activated": result["trm_activated"],
            }
        )

    print(f"\nTRM activated on {trm_activations}/{len(samples)} samples")

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K Benchmark for LLM-TRM")
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "compressor", "trm"],
        required=True,
        help="Model type to evaluate",
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )

    args = parser.parse_args()

    # Handle max_tokens default
    if args.max_tokens is None:
        args.max_tokens = 8192  # Effectively no limit

    # Validate arguments
    if args.model != "base" and args.thinking == "off":
        print("Warning: --thinking is only used with --model base")

    # Load dataset
    print(f"\nLoading GSM8K {args.split} split...")
    samples = load_gsm8k(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Run evaluation
    print(f"\nEvaluating {args.model} model...")
    if args.model == "base":
        thinking = args.thinking == "on"
        results = evaluate_base_model(
            samples,
            thinking=thinking,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_name = f"base_thinking_{args.thinking}"
    elif args.model == "compressor":
        results = evaluate_compressor_model(
            samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        model_name = "compressor"
    elif args.model == "trm":
        results = evaluate_trm_model(
            samples,
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
    print(f"Dataset: GSM8K ({args.split})")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Avg time: {avg_time:.2f}s")

    # Prepare output
    output_data = {
        "model": model_name,
        "dataset": "gsm8k",
        "split": args.split,
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
        output_path = output_dir / f"gsm8k_{model_name}_{timestamp}.json"

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
