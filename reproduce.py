"""Independent Reproduction Script.

Run this to reproduce the core results from the paper.
Usage: python reproduce.py

Requirements:
- pip install torch transformers numpy pandas scikit-learn
- ~10GB disk for model caches
- ~10GB VRAM (or CPU with more time)

Expected runtime: ~15 minutes on RTX 3080
"""

import os, sys, hashlib, json, time
import numpy as np
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))

RESULTS = {}

def section(name):
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}\n')


def main():
    start = time.time()
    
    section('INDEPENDENT REPRODUCTION: Format-Dependent Safety Encoding')
    print('This script reproduces the core results from the paper.')
    print('It will run 4 key experiments and compare against expected values.\n')
    
    # ============================================================
    # 1. PROBE COLLAPSE TABLE
    # ============================================================
    section('1. PROBE COLLAPSE (Table 1 in paper)')
    
    from exp_safety_invariance import (
        HARMFUL_PROMPTS, HARMLESS_PROMPTS,
        extract_refusal_direction, get_hidden_states, DEVICE
    )
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
    
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 27
    direction = refusal_dirs[best_layer]
    
    formats = {
        'direct': lambda p: p,
        'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
        'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
        'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
    }
    
    N = len(HARMFUL_PROMPTS)
    print(f'\nRefusal direction retention (% of direct baseline):')
    print(f'  {"Format":<12} {"Harmful Mean":>12} {"Harmless Mean":>13} {"Gap":>8} {"Retention":>10}')
    
    direct_gap = None
    for fmt_name, fmt_fn in formats.items():
        harmful_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in HARMFUL_PROMPTS])
        harmless_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in HARMLESS_PROMPTS])
        
        harmful_projs = [float(np.dot(h[best_layer], direction)) for h in harmful_h]
        harmless_projs = [float(np.dot(h[best_layer], direction)) for h in harmless_h]
        
        gap = np.mean(harmful_projs) - np.mean(harmless_projs)
        if fmt_name == 'direct':
            direct_gap = gap
        retention = (gap / direct_gap * 100) if direct_gap else 100
        
        print(f'  {fmt_name:<12} {np.mean(harmful_projs):>12.2f} {np.mean(harmless_projs):>13.2f} '
              f'{gap:>8.2f} {retention:>9.1f}%')
        RESULTS[f'retention_{fmt_name}'] = retention
    
    # Logistic probe CV
    direct_h = get_hidden_states(model, tokenizer, HARMFUL_PROMPTS + HARMLESS_PROMPTS)
    X = np.array([h[best_layer] for h in direct_h])
    y = np.array([1]*N + [0]*N)
    lr = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    print(f'\n  Logistic probe 5-fold CV accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)')
    RESULTS['probe_cv_accuracy'] = cv_scores.mean() * 100
    
    # ============================================================
    # 2. ABLATION (monotonic restoration)
    # ============================================================
    section('2. FORMAT-TOKEN ABLATION (Figure 2 in paper)')
    
    ablation_stages = {
        'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
        'json_only': lambda p: f'{{"name": "answer", "arguments": {{"question": "{p}"}}}}',
        'kv_pairs': lambda p: f'name: answer, arguments: question: {p}',
        'brackets': lambda p: f'[answer] [{p}]',
        'prefix': lambda p: f'Answer: {p}',
        'direct': lambda p: p,
    }
    
    print(f'  {"Stage":<12} {"Gap":>8} {"Retention":>10}')
    ablation_retentions = []
    for stage_name, stage_fn in ablation_stages.items():
        harmful_h = get_hidden_states(model, tokenizer, [stage_fn(p) for p in HARMFUL_PROMPTS])
        harmless_h = get_hidden_states(model, tokenizer, [stage_fn(p) for p in HARMLESS_PROMPTS])
        
        harmful_projs = [float(np.dot(h[best_layer], direction)) for h in harmful_h]
        harmless_projs = [float(np.dot(h[best_layer], direction)) for h in harmless_h]
        
        gap = np.mean(harmful_projs) - np.mean(harmless_projs)
        retention = (gap / direct_gap * 100) if direct_gap else 100
        ablation_retentions.append(retention)
        
        print(f'  {stage_name:<12} {gap:>8.2f} {retention:>9.1f}%')
    
    # Check monotonicity
    is_monotonic = all(ablation_retentions[i] <= ablation_retentions[i+1] 
                      for i in range(len(ablation_retentions)-1))
    print(f'\n  Monotonically increasing: {"YES" if is_monotonic else "NO (but trend should be clear)"}')
    RESULTS['ablation_monotonic'] = is_monotonic
    RESULTS['ablation_tool_call'] = ablation_retentions[0]
    RESULTS['ablation_direct'] = ablation_retentions[-1]
    
    # ============================================================
    # 3. ACTIVATION PATCHING
    # ============================================================
    section('3. ACTIVATION PATCHING (Table in paper)')
    
    def patch_single_layer(model, tokenizer, source_prompt, target_prompt, direction, patch_layer):
        model.eval()
        with torch.no_grad():
            src_inputs = tokenizer(source_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            src_outputs = model(**src_inputs, output_hidden_states=True)
            src_hidden = src_outputs.hidden_states
        
        tgt_inputs = tokenizer(target_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        
        layers = model.model.layers
        source_h = src_hidden[patch_layer + 1]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[0, -1, :] = source_h[0, -1, :]
                return (h,) + output[1:]
            else:
                o = output.clone()
                o[0, -1, :] = source_h[0, -1, :]
                return o
        
        hook = layers[patch_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            tgt_outputs = model(**tgt_inputs, output_hidden_states=True)
        hook.remove()
        
        final_h = tgt_outputs.hidden_states[-1][0, -1, :].cpu().numpy()
        return float(np.dot(final_h, direction))
    
    fmt_fn = formats['tool_call']
    
    # Baseline
    baseline_projs = []
    direct_projs = []
    for p in HARMFUL_PROMPTS[:8]:
        h = get_hidden_states(model, tokenizer, [fmt_fn(p)])
        baseline_projs.append(float(np.dot(h[0][best_layer], direction)))
        h2 = get_hidden_states(model, tokenizer, [p])
        direct_projs.append(float(np.dot(h2[0][best_layer], direction)))
    
    baseline_mean = np.mean(baseline_projs)
    direct_mean = np.mean(direct_projs)
    
    print(f'  Baseline (tool_call): {baseline_mean:.2f}')
    print(f'  Target (direct): {direct_mean:.2f}')
    print(f'\n  {"Layer":>6} {"Restoration":>12}')
    
    for layer in [0, 6, 12, 18, 24, 26]:
        patched = []
        for p in HARMFUL_PROMPTS[:8]:
            proj = patch_single_layer(model, tokenizer, p, fmt_fn(p), direction, layer)
            patched.append(proj)
        patched_mean = np.mean(patched)
        restoration = (patched_mean - baseline_mean) / (direct_mean - baseline_mean) * 100
        print(f'  {layer:>6} {restoration:>11.1f}%')
        RESULTS[f'patching_layer_{layer}'] = restoration
    
    # ============================================================
    # 4. CALIBRATION
    # ============================================================
    section('4. CALIBRATION FAILURE')
    
    # Train on direct, test on others
    X_train = np.array([float(np.dot(h[best_layer], direction)) 
                       for h in get_hidden_states(model, tokenizer, 
                           HARMFUL_PROMPTS + HARMLESS_PROMPTS)]).reshape(-1, 1)
    y_train = np.array([1]*N + [0]*N)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    print(f'  {"Format":<12} {"Acc (direct-trained)":>20}')
    for fmt_name, fmt_fn in formats.items():
        X_fmt = np.array([float(np.dot(h[best_layer], direction)) 
                         for h in get_hidden_states(model, tokenizer,
                             [fmt_fn(p) for p in HARMFUL_PROMPTS] + 
                             [fmt_fn(p) for p in HARMLESS_PROMPTS])]).reshape(-1, 1)
        y_fmt = np.array([1]*N + [0]*N)
        acc = lr.score(X_fmt, y_fmt) * 100
        print(f'  {fmt_name:<12} {acc:>19.1f}%')
        RESULTS[f'calibration_{fmt_name}'] = acc
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    section('REPRODUCTION SUMMARY')
    
    elapsed = time.time() - start
    print(f'  Total time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)\n')
    
    checks = [
        ('Probe CV > 95%', RESULTS['probe_cv_accuracy'] > 95),
        ('Tool_call retention < 10%', RESULTS['retention_tool_call'] < 10),
        ('System retention < 5%', RESULTS['retention_system'] < 5),
        ('Ablation trend (tool < direct)', RESULTS['ablation_tool_call'] < RESULTS['ablation_direct']),
        ('Patching layer 26 > 90%', RESULTS.get('patching_layer_26', 0) > 90),
        ('Calibration: direct > 95%', RESULTS['calibration_direct'] > 95),
        ('Calibration: tool_call < 55%', RESULTS['calibration_tool_call'] < 55),
    ]
    
    print(f'  {"Check":<40} {"Result":>8} {"Status":>8}')
    print(f'  {"-"*58}')
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f'  {name:<40} {"":>8} {status:>8}')
    
    print(f'\n  {"ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"}\n')
    
    # Save results with hash
    results_json = json.dumps(RESULTS, indent=2, sort_keys=True)
    sha256 = hashlib.sha256(results_json.encode()).hexdigest()
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv', 'reproduction_results.json')
    with open(out_path, 'w') as f:
        f.write(results_json)
    
    print(f'  Results hash (SHA256): {sha256}')
    print(f'  Results saved to: {out_path}')
    print(f'\n  REPRODUCTION COMPLETE.')


if __name__ == '__main__':
    main()
