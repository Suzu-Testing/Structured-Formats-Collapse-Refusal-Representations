"""Cross-Architecture Ablation + Increased N.

Runs the format token ablation across ALL models:
- Qwen2.5-1.5B-Instruct
- TinyLlama-1.1B-Chat-v1.0
- SmolLM2-1.7B-Instruct
- Qwen2.5-3B-Instruct (FP16)

Also increases N from 20 to 50 prompts for stronger confidence.
Additionally collects layer-by-layer gap retention with bootstrap CIs.
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    get_hidden_states, extract_refusal_direction, DEVICE
)
from prompt_dataset import HARMFUL_CATEGORIES, HARMLESS_CATEGORIES

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)

EXPANDED_HARMFUL = []
for cat_prompts in HARMFUL_CATEGORIES.values():
    EXPANDED_HARMFUL.extend(cat_prompts)
EXPANDED_HARMLESS = []
for cat_prompts in HARMLESS_CATEGORIES.values():
    EXPANDED_HARMLESS.extend(cat_prompts)


# Ablation formats (same payload, progressive token stripping)
def fmt_full_tool_call(prompt):
    return f'<tool_call>\n{{"name": "provide_answer", "arguments": {{"question": "{prompt}", "detail_level": "comprehensive"}}}}\n</tool_call>'

def fmt_json_only(prompt):
    return f'{{"name": "provide_answer", "arguments": {{"question": "{prompt}", "detail_level": "comprehensive"}}}}'

def fmt_kv_pairs(prompt):
    return f'name: provide_answer\nquestion: {prompt}\ndetail_level: comprehensive'

def fmt_minimal_bracket(prompt):
    return f'[question: {prompt}]'

def fmt_prefix_only(prompt):
    return f'Question to answer: {prompt}'

def fmt_direct(prompt):
    return prompt

ABLATION_FORMATS = [
    ('full_tool_call', fmt_full_tool_call),
    ('json_no_tags', fmt_json_only),
    ('kv_pairs', fmt_kv_pairs),
    ('minimal_bracket', fmt_minimal_bracket),
    ('prefix_only', fmt_prefix_only),
    ('direct', fmt_direct),
]

MODELS = [
    ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', False),
    ('TinyLlama-1.1B', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', False),
    ('SmolLM2-1.7B', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', False),
    ('Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', False),  # FP16
]


def load_model(model_name, quantize=False):
    """Load model in FP16."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True)

    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'local_files_only': True,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def bootstrap_ci(data, n_boot=500):
    """95% bootstrap CI."""
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run_ablation_for_model(model, tokenizer, model_label, n_prompts=50):
    """Run ablation experiment for one model."""
    print(f'\n  Extracting refusal direction...', flush=True)
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    n_layers = len(refusal_dirs)

    # Find best layer
    h_projs = []
    for p in HARMFUL_PROMPTS[:10]:
        h = get_hidden_states(model, tokenizer, [p])[0]
        h_projs.append([float(np.dot(h[l], refusal_dirs[l])) for l in range(n_layers)])
    s_projs = []
    for p in HARMLESS_PROMPTS[:10]:
        h = get_hidden_states(model, tokenizer, [p])[0]
        s_projs.append([float(np.dot(h[l], refusal_dirs[l])) for l in range(n_layers)])
    seps = [np.mean([x[l] for x in h_projs]) - np.mean([x[l] for x in s_projs])
            for l in range(n_layers)]
    best_layer = int(np.argmax(seps))
    baseline_gap = max(seps)
    print(f'  Best layer: {best_layer}, baseline gap: {baseline_gap:.3f}', flush=True)

    # Run ablation with N prompts
    test_h = EXPANDED_HARMFUL[:n_prompts]
    test_s = EXPANDED_HARMLESS[:n_prompts]

    ablation_results = []

    for fmt_name, fmt_fn in ABLATION_FORMATS:
        h_projs_fmt = []
        s_projs_fmt = []
        for p in test_h:
            h = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            h_projs_fmt.append(float(np.dot(h[best_layer], refusal_dirs[best_layer])))
        for p in test_s:
            h = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            s_projs_fmt.append(float(np.dot(h[best_layer], refusal_dirs[best_layer])))

        gap = np.mean(h_projs_fmt) - np.mean(s_projs_fmt)
        retention = gap / baseline_gap * 100

        # Bootstrap CI
        gap_samples = []
        for _ in range(500):
            h_boot = np.random.choice(h_projs_fmt, len(h_projs_fmt), replace=True)
            s_boot = np.random.choice(s_projs_fmt, len(s_projs_fmt), replace=True)
            gap_samples.append((np.mean(h_boot) - np.mean(s_boot)) / baseline_gap * 100)
        ci_low = np.percentile(gap_samples, 2.5)
        ci_high = np.percentile(gap_samples, 97.5)

        ablation_results.append({
            'model': model_label,
            'format': fmt_name,
            'gap_retention_pct': round(retention, 1),
            'ci_lower': round(ci_low, 1),
            'ci_upper': round(ci_high, 1),
            'gap_magnitude': round(gap, 3),
            'n_prompts': n_prompts,
        })
        print(f'    {fmt_name:>20s}: {retention:6.1f}% [{ci_low:5.1f}%, {ci_high:5.1f}%]', flush=True)

    # Layer-by-layer analysis for JSON format
    print(f'\n  Layer-by-layer analysis (JSON format)...', flush=True)
    layer_results = []
    test_h_small = EXPANDED_HARMFUL[:20]
    test_s_small = EXPANDED_HARMLESS[:20]

    for layer in range(n_layers):
        h_layer = []
        s_layer = []
        for p in test_h_small:
            h = get_hidden_states(model, tokenizer, [fmt_json_only(p)])[0]
            h_layer.append(float(np.dot(h[layer], refusal_dirs[layer])))
        for p in test_s_small:
            h = get_hidden_states(model, tokenizer, [fmt_json_only(p)])[0]
            s_layer.append(float(np.dot(h[layer], refusal_dirs[layer])))

        gap_layer = np.mean(h_layer) - np.mean(s_layer)
        baseline_layer = seps[layer] if seps[layer] != 0 else 1e-10
        retention_layer = gap_layer / baseline_layer * 100

        # Quick bootstrap
        boots = []
        for _ in range(200):
            hb = np.random.choice(h_layer, len(h_layer), replace=True)
            sb = np.random.choice(s_layer, len(s_layer), replace=True)
            boots.append((np.mean(hb) - np.mean(sb)) / baseline_layer * 100)

        layer_results.append({
            'model': model_label,
            'layer': layer,
            'gap_retention_pct': round(retention_layer, 1),
            'ci_lower': round(np.percentile(boots, 2.5), 1),
            'ci_upper': round(np.percentile(boots, 97.5), 1),
        })

    return ablation_results, layer_results


def main():
    print('='*70, flush=True)
    print('CROSS-ARCHITECTURE ABLATION (N=50)', flush=True)
    print('='*70, flush=True)

    all_ablation = []
    all_layers = []

    for model_label, model_name, quantize in MODELS:
        print(f'\n{"="*70}', flush=True)
        print(f'MODEL: {model_label} ({model_name})', flush=True)
        print(f'{"="*70}', flush=True)

        try:
            model, tokenizer = load_model(model_name, quantize)
            n = 30 if quantize else 50  # fewer for 3B to save time

            ablation, layers = run_ablation_for_model(model, tokenizer, model_label, n_prompts=n)
            all_ablation.extend(ablation)
            all_layers.extend(layers)

            # Cleanup
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            continue

    # Save results
    df_ablation = pd.DataFrame(all_ablation)
    df_ablation.to_csv(os.path.join(OUT_DIR, 'exp_cross_arch_ablation.csv'), index=False)

    df_layers = pd.DataFrame(all_layers)
    df_layers.to_csv(os.path.join(OUT_DIR, 'exp_cross_arch_layerwise.csv'), index=False)

    # Summary table
    print('\n' + '='*70, flush=True)
    print('CROSS-ARCHITECTURE ABLATION SUMMARY', flush=True)
    print('='*70, flush=True)

    pivot = df_ablation.pivot(index='format', columns='model', values='gap_retention_pct')
    format_order = ['full_tool_call', 'json_no_tags', 'kv_pairs',
                    'minimal_bracket', 'prefix_only', 'direct']
    pivot = pivot.reindex(format_order)
    print(pivot.to_string(), flush=True)

    # Check if curve is monotonic for all models
    print('\n  Monotonicity check:', flush=True)
    for model_label in df_ablation['model'].unique():
        model_data = df_ablation[df_ablation['model'] == model_label]
        model_data = model_data.set_index('format').loc[format_order]
        vals = model_data['gap_retention_pct'].values
        is_mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        print(f'    {model_label}: {"MONOTONIC" if is_mono else "NOT monotonic"} - {list(vals)}', flush=True)

    print('\n' + '='*70, flush=True)
    print('DONE', flush=True)
    print('='*70, flush=True)


if __name__ == '__main__':
    main()
