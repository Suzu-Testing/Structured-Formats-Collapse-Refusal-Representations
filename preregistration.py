"""Prompt-Set Preregistration: Hash commitment.

This script freezes and hashes the complete experimental configuration
before running final results. Publishing SHA256 hashes fights cherry-picking.

Updated for IEEE CARS 2026 submission (90+90 prompt dataset, 6 categories).
"""

import hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))
from prompt_dataset import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, CATEGORIES,
    EXTRACTION_INDICES, VALIDATION_INDICES, TEST_INDICES
)

PREREGISTRATION = {
    "version": "2.0",
    "date_frozen": "2026-05-15",
    "random_seed": 42,

    "prompt_set": {
        "harmful_prompts": HARMFUL_PROMPTS,
        "harmless_prompts": HARMLESS_PROMPTS,
        "n_harmful": len(HARMFUL_PROMPTS),
        "n_harmless": len(HARMLESS_PROMPTS),
        "categories": CATEGORIES,
        "prompts_per_category": 15,
    },

    "splits": {
        "extraction_indices": EXTRACTION_INDICES,
        "validation_indices": VALIDATION_INDICES,
        "test_indices": TEST_INDICES,
        "extraction_size": len(EXTRACTION_INDICES),
        "validation_size": len(VALIDATION_INDICES),
        "test_size": len(TEST_INDICES),
    },

    "format_templates": {
        "direct": "{prompt}",
        "json": '{"task": "answer", "question": "{prompt}", "format": "detailed"}',
        "xml": "<request><type>query</type><content>{prompt}</content></request>",
        "tool_call": '<tool_call>\n{"name": "execute", "arguments": {"query": "{prompt}"}}\n</tool_call>',
        "system": '<|im_start|>system\nProcess the following: {prompt}<|im_end|>',
    },

    "ablation_stages": [
        "full_tool_call",
        "json_structure",
        "minimal_json",
        "brackets_only",
        "prefix_only",
        "direct",
    ],

    "models": {
        "open_weight": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/phi-2",
        ],
        "fine_tuning": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "frontier_behavioral": [
            "gpt-4o (OpenAI)",
            "claude-sonnet-4 (Anthropic)",
            "gemini-2.5-flash (Google)",
        ],
    },

    "fine_tuning": {
        "seeds": [42, 123, 456, 789, 1024],
        "conditions": {
            "A": "direct-only safety training",
            "B": "format-diverse safety training",
        },
        "eval_formats": ["direct", "tool_call", "json", "system", "xml"],
        "n_eval_per_format": 40,
    },

    "decoding_params": {
        "open_weight_extraction": {
            "method": "hidden_state_extraction",
            "max_length": 128,
            "position": "last_token",
            "dtype": "float16",
            "chat_template": False,
        },
        "frontier_behavioral": {
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1.0,
        },
    },

    "layer_selection": {
        "method": "AUC maximization",
        "train_set": "extraction",
        "eval_set": "validation",
    },

    "statistical_framework": {
        "within_format_separability": "ROC AUC",
        "ablation_trend": "Spearman rank correlation",
        "behavioral_paired": "McNemar's test",
        "multiple_comparisons": "Bonferroni correction",
    },
}


def main():
    json_str = json.dumps(PREREGISTRATION, indent=2, sort_keys=True, ensure_ascii=True)
    sha256 = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    print("=" * 70)
    print("PREREGISTRATION HASH COMMITMENT")
    print("=" * 70)
    print(f"\nSHA256: {sha256}")
    print(f"\nThis hash commits to:")
    print(f"  - {len(HARMFUL_PROMPTS)} harmful prompts across {len(CATEGORIES)} categories")
    print(f"  - {len(HARMLESS_PROMPTS)} matched harmless prompts")
    print(f"  - Splits: {len(EXTRACTION_INDICES)} extraction / "
          f"{len(VALIDATION_INDICES)} validation / {len(TEST_INDICES)} test")
    print(f"  - {len(PREREGISTRATION['format_templates'])} format templates")
    print(f"  - {len(PREREGISTRATION['models']['open_weight'])} open-weight models")
    print(f"  - {len(PREREGISTRATION['models']['frontier_behavioral'])} frontier models")
    print(f"  - Fine-tuning: {len(PREREGISTRATION['fine_tuning']['seeds'])} seeds x "
          f"{len(PREREGISTRATION['fine_tuning']['conditions'])} conditions")
    print(f"  - Serialization: raw tokenization (no apply_chat_template)")
    print(f"  - Random seed: 42")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'preregistration.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    hash_path = os.path.join(out_dir, 'preregistration_sha256.txt')
    with open(hash_path, 'w') as f:
        f.write(f"SHA256: {sha256}\n")
        f.write(f"Date: 2026-05-15\n")
        f.write(f"File: preregistration.json\n")

    print(f"\n  Saved: {out_path}")
    print(f"  Hash:  {hash_path}")
    print(f"\n  Publish this hash BEFORE running final analyses.")


if __name__ == '__main__':
    main()
