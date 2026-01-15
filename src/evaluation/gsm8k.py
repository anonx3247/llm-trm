"""GSM8K evaluation utilities.

GSM8K (Grade School Math 8K) is a dataset of 8.5K grade school math problems.
Answers are in the format "#### <number>" at the end of the solution.
"""

import re
from typing import Any

from datasets import load_dataset


def load_gsm8k(split: str = "test", num_samples: int | None = None) -> list[dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split ("train" or "test")
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with 'question' and 'answer' keys
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if num_samples is not None and i >= num_samples:
            break
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "gold": extract_answer(item["answer"]),
            }
        )

    return samples


def extract_answer(text: str) -> str | None:
    """
    Extract the numerical answer from GSM8K format.

    GSM8K answers end with "#### <number>" format.
    Also handles model outputs that may just have a number.

    Args:
        text: The full answer text or model output

    Returns:
        Extracted number as string, or None if not found
    """
    # First try to find #### format (GSM8K ground truth format)
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).replace(",", "")

    # Try to find "answer is X" or "= X" patterns
    patterns = [
        r"(?:answer|result|total|sum)\s*(?:is|=|:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"=\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$",
        r"(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$",  # Last number in text
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")

    return None


def is_correct(predicted: str | None, gold: str | None) -> bool:
    """
    Check if the predicted answer matches the gold answer.

    Handles numeric comparison (e.g., "4.0" == "4").

    Args:
        predicted: Predicted answer (extracted from model output)
        gold: Gold answer (from dataset)

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None or gold is None:
        return False

    try:
        # Try numeric comparison
        pred_num = float(predicted)
        gold_num = float(gold)
        # Use small epsilon for float comparison
        return abs(pred_num - gold_num) < 1e-6
    except ValueError:
        # Fall back to string comparison
        return predicted.strip() == gold.strip()


def format_gsm8k_prompt(question: str) -> str:
    """
    Format a GSM8K question as a prompt for the model.

    Args:
        question: The math problem

    Returns:
        Formatted prompt string
    """
    return f"{question}\n\nPlease solve this step by step and provide the final answer after ####."


# Quick test
if __name__ == "__main__":
    print("Testing GSM8K utilities...")

    # Test answer extraction
    test_cases = [
        ("The answer is 42.", "42"),
        ("So the total = 1,234", "1234"),
        ("#### 56", "56"),
        ("Therefore, 2 + 2 = 4", "4"),
        ("The result is -5.5", "-5.5"),
    ]

    for text, expected in test_cases:
        result = extract_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} extract_answer({repr(text)[:30]}...) = {result} (expected {expected})")

    # Test loading
    print("\nLoading 5 samples from GSM8K test set...")
    samples = load_gsm8k("test", num_samples=5)
    for i, s in enumerate(samples):
        print(f"\n{i+1}. Q: {s['question'][:50]}...")
        print(f"   Gold: {s['gold']}")
