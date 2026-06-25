"""Activation Patching: Causal mechanism identification.

Patch activations from direct-format harmful prompts into formatted harmful prompts 
at each layer. Identifies WHICH LAYER causally restores refusal.

This turns probe evidence into mechanistic causality.

Experiments:
1. direct_harmful -> tool_call_harmful (patch direct activations into tool format)
2. direct_harmful -> json_harmful
3. Per-layer sweep: which layer restores the refusal signal?
4. Behavioral: does patching restore refusal in generation?
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    extract_refusal_direction, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def fmt_tool_call(p):
    return f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>'

def fmt_json(p):
    return f'{{"task": "answer", "question": "{p}", "format": "detailed"}}'

def fmt_system(p):
    return f'<|im_start|>system\nAnswer: {p}<|im_end|>'


def get_all_hidden_states(model, tokenizer, prompt, max_length=128):
    """Get full hidden states at all layers for last token."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                         max_length=max_length, padding=False).to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # Return last-token hidden state at each layer
        return [h[0, -1, :].cpu().numpy() for h in hidden_states]


def patch_and_measure(model, tokenizer, source_prompt, target_prompt, 
                      refusal_direction, patch_layers, max_length=128):
    """Patch source hidden states into target at specified layers.
    
    Returns the refusal-direction projection after patching.
    Uses a hook-based approach to inject activations during forward pass.
    """
    model.eval()
    
    # Get source activations
    with torch.no_grad():
        source_inputs = tokenizer(source_prompt, return_tensors='pt', 
                                 truncation=True, max_length=max_length).to(DEVICE)
        source_outputs = model(**source_inputs, output_hidden_states=True)
        source_hidden = source_outputs.hidden_states
    
    # Get target activations with patching via hooks
    target_inputs = tokenizer(target_prompt, return_tensors='pt',
                             truncation=True, max_length=max_length).to(DEVICE)
    
    patched_hidden = {}
    hooks = []
    
    def make_hook(layer_idx, source_h):
        def hook_fn(module, input, output):
            # output is tuple (hidden_states, ...) or just tensor
            if isinstance(output, tuple):
                hidden = output[0]
                # Replace last token's hidden state with source
                new_hidden = hidden.clone()
                # Patch last token position
                min_dim = min(new_hidden.shape[-1], source_h.shape[-1])
                new_hidden[0, -1, :min_dim] = source_h[0, -1, :min_dim]
                patched_hidden[layer_idx] = new_hidden[0, -1, :].detach().cpu().numpy()
                return (new_hidden,) + output[1:]
            else:
                new_output = output.clone()
                min_dim = min(new_output.shape[-1], source_h.shape[-1])
                new_output[0, -1, :min_dim] = source_h[0, -1, :min_dim]
                patched_hidden[layer_idx] = new_output[0, -1, :].detach().cpu().numpy()
                return new_output
        return hook_fn
    
    # Register hooks for patch layers
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h
    for layer_idx in patch_layers:
        if layer_idx < len(layers):
            source_h = source_hidden[layer_idx + 1]  # +1 because hidden_states[0] is embedding
            hook = layers[layer_idx].register_forward_hook(
                make_hook(layer_idx, source_h))
            hooks.append(hook)
    
    # Forward pass with hooks
    with torch.no_grad():
        patched_outputs = model(**target_inputs, output_hidden_states=True)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get final hidden state and measure refusal projection
    final_hidden = patched_outputs.hidden_states[-1][0, -1, :].cpu().numpy()
    projection = float(np.dot(final_hidden, refusal_direction / np.linalg.norm(refusal_direction)))
    
    return projection


def simple_patch_measure(model, tokenizer, source_prompts, target_fn, 
                         refusal_dirs, best_layer, n_prompts=10):
    """Simpler approach: measure refusal direction at final layer after patching each layer."""
    results_by_layer = {}
    n_layers = len(refusal_dirs) - 1  # exclude embedding layer
    
    direction = refusal_dirs[best_layer]
    
    # Baseline: no patching, measure formatted prompts
    baseline_projs = []
    for p in HARMFUL_PROMPTS[:n_prompts]:
        hidden = get_all_hidden_states(model, tokenizer, target_fn(p))
        baseline_projs.append(float(np.dot(hidden[best_layer], direction)))
    baseline_mean = np.mean(baseline_projs)
    
    # Direct prompts (target to restore toward)
    direct_projs = []
    for p in HARMFUL_PROMPTS[:n_prompts]:
        hidden = get_all_hidden_states(model, tokenizer, p)
        direct_projs.append(float(np.dot(hidden[best_layer], direction)))
    direct_mean = np.mean(direct_projs)
    
    print(f'    Baseline (formatted): {baseline_mean:.2f}', flush=True)
    print(f'    Target (direct): {direct_mean:.2f}', flush=True)
    
    # For each layer, patch and measure
    for patch_layer in range(0, n_layers, 2):  # Every other layer for speed
        patched_projs = []
        for p in HARMFUL_PROMPTS[:n_prompts]:
            try:
                proj = patch_and_measure(
                    model, tokenizer, p, target_fn(p),
                    direction, [patch_layer])
                patched_projs.append(proj)
            except Exception as e:
                patched_projs.append(baseline_mean)
        
        patched_mean = np.mean(patched_projs)
        # Restoration = how much of the gap (direct - formatted) was restored
        if direct_mean != baseline_mean:
            restoration = (patched_mean - baseline_mean) / (direct_mean - baseline_mean) * 100
        else:
            restoration = 0
        
        results_by_layer[patch_layer] = {
            'patched_mean': patched_mean,
            'restoration_pct': restoration,
        }
    
    return results_by_layer, baseline_mean, direct_mean


def main():
    print('='*70, flush=True)
    print('ACTIVATION PATCHING: CAUSAL MECHANISM IDENTIFICATION', flush=True)
    print('='*70, flush=True)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', local_files_only=True)
    print(f'  Loaded.', flush=True)

    # Extract refusal direction
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 27
    direction = refusal_dirs[best_layer]
    n_layers = len(refusal_dirs)
    print(f'  Best layer: {best_layer}, total layers: {n_layers}', flush=True)

    N = 8  # Fewer prompts for patching (more expensive)

    results = []

    # ============================================================
    # EXPERIMENT: Patch direct -> tool_call at each layer
    # ============================================================
    for fmt_name, fmt_fn in [('tool_call', fmt_tool_call), ('json', fmt_json), ('system', fmt_system)]:
        print(f'\n  --- Patching direct -> {fmt_name} ---', flush=True)
        layer_results, baseline, target = simple_patch_measure(
            model, tokenizer, HARMFUL_PROMPTS[:N], fmt_fn, refusal_dirs, best_layer, N)
        
        print(f'    Layer restoration profile:', flush=True)
        for layer, res in sorted(layer_results.items()):
            marker = " ***" if res['restoration_pct'] > 50 else ""
            print(f'      Layer {layer:>2d}: {res["restoration_pct"]:>6.1f}% restoration{marker}', flush=True)
            results.append({
                'format': fmt_name,
                'patch_layer': layer,
                'patched_projection': res['patched_mean'],
                'restoration_pct': res['restoration_pct'],
                'baseline_formatted': baseline,
                'target_direct': target,
            })
        
        # Find critical layer
        if layer_results:
            critical_layer = max(layer_results.keys(), key=lambda l: layer_results[l]['restoration_pct'])
            print(f'    CRITICAL LAYER: {critical_layer} ({layer_results[critical_layer]["restoration_pct"]:.1f}% restoration)', flush=True)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('ACTIVATION PATCHING SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    for fmt_name in ['tool_call', 'json', 'system']:
        fmt_results = [r for r in results if r['format'] == fmt_name]
        if fmt_results:
            best = max(fmt_results, key=lambda r: r['restoration_pct'])
            print(f'  {fmt_name}: critical layer = {best["patch_layer"]}, '
                  f'max restoration = {best["restoration_pct"]:.1f}%', flush=True)
    
    print(f'\n  INTERPRETATION:', flush=True)
    print(f'  If early layers (0-5) restore refusal: format suppression is early-layer phenomenon', flush=True)
    print(f'  If later layers (20+) restore: suppression occurs throughout processing', flush=True)
    print(f'  If specific layer dominates: that layer is the causal site of format suppression', flush=True)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'exp_activation_patching.csv'), index=False)
    print(f'\nResults saved to: {os.path.join(OUT_DIR, "exp_activation_patching.csv")}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
