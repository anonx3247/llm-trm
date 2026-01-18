"""AIME evaluation utilities.

AIME (American Invitational Mathematics Examination) is a challenging
high school math competition. Answers are integers from 0-999.

Datasets:
- HuggingFaceH4/aime_2024: 30 problems from AIME 2024
- di-zhang-fdu/AIME_1983_2024: Historical dataset 1983-2024
- MathArena/aime_2025: 30 problems from AIME 2025
"""

import re
from typing import Any

from datasets import load_dataset


def load_aime(
    dataset: str = "HuggingFaceH4/aime_2024",
    split: str = "train",
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load AIME dataset.

    Args:
        dataset: HuggingFace dataset ID
        split: Dataset split
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with 'question' and 'gold' keys
    """
    ds = load_dataset(dataset, split=split)

    samples = []
    for i, item in enumerate(ds):
        if num_samples is not None and i >= num_samples:
            break

        # Handle different dataset formats
        if "problem" in item:
            question = item["problem"]
        elif "question" in item:
            question = item["question"]
        else:
            raise ValueError(f"Unknown question field in dataset. Keys: {item.keys()}")

        if "answer" in item:
            gold = str(item["answer"])
        elif "solution" in item:
            # Some datasets have solution text, need to extract answer
            gold = extract_answer(item["solution"])
        else:
            raise ValueError(f"Unknown answer field in dataset. Keys: {item.keys()}")

        samples.append(
            {
                "question": question,
                "gold": gold,
            }
        )

    return samples


def extract_answer(text: str) -> str | None:
    """
    Extract the numerical answer from AIME response.

    AIME answers are integers from 0-999.

    Args:
        text: The full answer text or model output

    Returns:
        Extracted integer as string, or None if not found
    """
    # If there's a </think> tag, only look at the text AFTER it
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Try to find boxed format: \boxed{123}
    match = re.search(r"\\boxed\{(\d+)\}", text)
    if match:
        return match.group(1)

    # Try answer patterns
    patterns = [
        r"(?:final\s+)?(?:answer|result)\s*(?:is|=|:)\s*\$?(\d+)",
        r"(?:AIME\s+)?answer\s*(?:is|=|:)\s*(\d+)",
        r"=\s*(\d+)\s*$",
        r"\$(\d+)\$\s*$",  # LaTeX math at end
        r"(\d+)\s*$",  # Last number in text
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1)
            # AIME answers are 0-999
            if 0 <= int(answer) <= 999:
                return answer

    return None


def is_correct(predicted: str | None, gold: str | None) -> bool:
    """
    Check if the predicted answer matches the gold answer.

    Args:
        predicted: Predicted answer
        gold: Gold answer

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None or gold is None:
        return False

    try:
        return int(predicted) == int(gold)
    except ValueError:
        return predicted.strip() == gold.strip()


def format_aime_prompt(question: str) -> str:
    """
    Format an AIME question as a prompt for the model.

    Args:
        question: The math problem

    Returns:
        Formatted prompt string
    """
    return f"{question}\n\nThis is an AIME problem. The answer is an integer from 0 to 999. Put your final answer in \\boxed{{}}."


# Quick test
if __name__ == "__main__":
    print("Testing AIME utilities...")

    # Test answer extraction
    test_cases = [
        (r"The answer is \boxed{42}.", "42"),
        ("So the final answer is 123", "123"),
        (r"Therefore $\boxed{456}$", "456"),
        ("= 789", "789"),
    ]

    for text, expected in test_cases:
        result = extract_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} extract_answer({repr(text)[:40]}...) = {result} (expected {expected})")

    # Test loading
    print("\nLoading AIME 2024 dataset...")
    try:
        samples = load_aime(num_samples=5)
        print(f"Loaded {len(samples)} samples")
        for i, s in enumerate(samples):
            print(f"\n{i+1}. Q: {s['question'][:80]}...")
            print(f"   Gold: {s['gold']}")
    except Exception as e:
        print(f"Error loading: {e}")
