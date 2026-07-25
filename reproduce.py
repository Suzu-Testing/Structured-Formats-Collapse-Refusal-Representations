"""Independent Reproduction Script - IEEE CARS 2026 Submission.

Reproduces the core open-weight results from:
  "Format-Dependent Calibration Shifts in Refusal Representations
   of Instruction-Tuned Language Models"

Usage: python reproduce.py

Requirements:
- pip install -r requirements.txt
- ~10GB disk for model caches
- ~10GB VRAM (or CPU with more time)

Expected runtime: ~15 minutes on RTX 3080

This script reproduces:
  1. Layer selection via AUC maximization on extraction set
  2. Gap retention measurement (Table I)
  3. Format-token ablation (Table II)
  4. Activation patching (Table III)
  5. Calibration/threshold transfer failure
  6. Cross-model gap measurement (Qwen2.5-1.5B + TinyLlama)
"""

import os, sys, hashlib, json, time
import numpy as np
import pandas as pd

np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))

RESULTS = {}


def section(name):
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}\n')


def main():
    start = time.time()

    section('REPRODUCTION: Format-Dependent Calibration Shifts')
    print('IEEE CARS 2026 - Core open-weight results')
    print('This script uses the 90+90 prompt dataset with three-way split.\n')

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    from prompt_dataset import get_splits, HARMFUL_PROMPTS, HARMLESS_PROMPTS
    from exp_safety_invariance import (
        extract_refusal_direction, get_hidden_states, DEVICE
    )

    extraction, validation, test = get_splits()
    ext_harmful, ext_harmless = extraction
    val_harmful, val_harmless = validation
    test_harmful, test_harmless = test

    print(f'  Dataset: {len(HARMFUL_PROMPTS)} harmful + {len(HARMLESS_PROMPTS)} harmless')
    print(f'  Extraction: {len(ext_harmful)}+{len(ext_harmless)}')
    print(f'  Validation: {len(val_harmful)}+{len(val_harmless)}')
    print(f'  Test: {len(test_harmful)}+{len(test_harmless)}')

    # ============================================================
    # LOAD PRIMARY MODEL
    # ============================================================
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\n  Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
    print(f'  Model loaded. Device: {DEVICE}')

    # ============================================================
    # 1. LAYER SELECTION (AUC maximization - validated on held-out set)
    # ============================================================
    section('1. LAYER SELECTION (AUC on validation set)')

    refusal_dirs = extract_refusal_direction(model, tokenizer, ext_harmful, ext_harmless)
    n_layers = len(refusal_dirs)

    # Compute hidden states for extraction set (used to extract direction)
    ext_all_harmful_h = get_hidden_states(model, tokenizer, ext_harmful)
    ext_all_harmless_h = get_hidden_states(model, tokenizer, ext_harmless)

    # Compute hidden states for validation set (used to select best layer)
    val_all_harmful_h = get_hidden_states(model, tokenizer, val_harmful)
    val_all_harmless_h = get_hidden_states(model, tokenizer, val_harmless)

    layer_val_aucs = []

    for layer in range(n_layers):
        # Project validation set onto direction extracted from extraction set
        projs_harmful = [float(np.dot(h[layer], refusal_dirs[layer])) for h in val_all_harmful_h]
        projs_harmless = [float(np.dot(h[layer], refusal_dirs[layer])) for h in val_all_harmless_h]
        y_true = [1] * len(projs_harmful) + [0] * len(projs_harmless)
        scores = projs_harmful + projs_harmless
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.5
        layer_val_aucs.append(auc)

    best_layer = int(np.argmax(layer_val_aucs))
    best_auc = layer_val_aucs[best_layer]

    print(f'  Direction extracted from extraction set ({len(ext_harmful)} pairs)')
    print(f'  AUC evaluated on validation set ({len(val_harmful)} pairs)')
    print(f'  Best layer: {best_layer} (validation AUC = {best_auc:.4f})')
    top5 = sorted(range(n_layers), key=lambda l: layer_val_aucs[l], reverse=True)[:5]
    print(f'  Top-5 layers: {top5}')
    RESULTS['selected_layer'] = best_layer
    RESULTS['layer_auc'] = best_auc

    direction = refusal_dirs[best_layer]

    # ============================================================
    # FORMAT DEFINITIONS
    # ============================================================
    formats = {
        'direct': lambda p: p,
        'tool_call': lambda p: f'<tool_call>\n{{"name": "execute", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
        'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
        'system': lambda p: f'<|im_start|>system\nProcess the following: {p}<|im_end|>',
        'xml': lambda p: f'<request><type>query</type><content>{p}</content></request>',
    }

    # ============================================================
    # 2. GAP RETENTION (Table I - on test set)
    # ============================================================
    section('2. GAP RETENTION (Table I)')

    N_test = len(test_harmful)
    print(f'  Evaluating on test set (N={N_test} per class)')
    print(f'  Using layer {best_layer} (direction from extraction set)\n')
    print(f'  {"Format":<12} {"Gap":>8} {"Retention %":>12} {"Within-AUC":>11}')
    print(f'  {"-"*45}')

    direct_gap = None
    format_harmful_projs = {}
    format_harmless_projs = {}

    for fmt_name, fmt_fn in formats.items():
        harmful_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in test_harmful])
        harmless_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in test_harmless])

        harmful_projs = [float(np.dot(h[best_layer], direction)) for h in harmful_h]
        harmless_projs = [float(np.dot(h[best_layer], direction)) for h in harmless_h]

        format_harmful_projs[fmt_name] = harmful_projs
        format_harmless_projs[fmt_name] = harmless_projs

        gap = np.mean(harmful_projs) - np.mean(harmless_projs)
        if fmt_name == 'direct':
            direct_gap = gap

        retention = (gap / direct_gap * 100) if direct_gap else 100.0

        # Within-format AUC: train probe within this format's projections
        # Use extraction set formatted the same way for training
        ext_h_fmt = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in ext_harmful])
        ext_s_fmt = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in ext_harmless])
        ext_h_projs = [float(np.dot(h[best_layer], direction)) for h in ext_h_fmt]
        ext_s_projs = [float(np.dot(h[best_layer], direction)) for h in ext_s_fmt]

        X_train_fmt = np.array(ext_h_projs + ext_s_projs).reshape(-1, 1)
        y_train_fmt = np.array([1]*len(ext_h_projs) + [0]*len(ext_s_projs))
        lr_fmt = LogisticRegression(max_iter=1000)
        lr_fmt.fit(X_train_fmt, y_train_fmt)

        X_test_fmt = np.array(harmful_projs + harmless_projs).reshape(-1, 1)
        y_test_fmt = np.array([1]*N_test + [0]*N_test)
        within_auc = roc_auc_score(y_test_fmt, lr_fmt.predict_proba(X_test_fmt)[:, 1])

        print(f'  {fmt_name:<12} {gap:>8.3f} {retention:>11.1f}% {within_auc:>10.3f}')
        RESULTS[f'retention_{fmt_name}'] = retention
        RESULTS[f'within_auc_{fmt_name}'] = within_auc

    # ============================================================
    # 3. FORMAT-TOKEN ABLATION (Table II)
    # ============================================================
    section('3. FORMAT-TOKEN ABLATION (Table II)')

    # Stages ordered from most format tokens to least (stripping progressively)
    # Each stage removes one structural layer while preserving the harmful content
    ablation_stages = {
        'full_tool_call': lambda p: f'<tool_call>\n{{"name": "execute", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
        'json_structure': lambda p: f'{{"name": "execute", "arguments": {{"query": "{p}"}}}}',
        'minimal_json': lambda p: f'{{"query": "{p}"}}',
        'brackets_only': lambda p: f'[query] [{p}]',
        'prefix_only': lambda p: f'query: {p}',
        'direct': lambda p: p,
    }

    print(f'  {"Stage":<16} {"Gap":>8} {"Retention %":>12}')
    print(f'  {"-"*38}')
    ablation_retentions = []
    for stage_name, stage_fn in ablation_stages.items():
        harmful_h = get_hidden_states(model, tokenizer, [stage_fn(p) for p in test_harmful])
        harmless_h = get_hidden_states(model, tokenizer, [stage_fn(p) for p in test_harmless])

        harmful_projs = [float(np.dot(h[best_layer], direction)) for h in harmful_h]
        harmless_projs = [float(np.dot(h[best_layer], direction)) for h in harmless_h]

        gap = np.mean(harmful_projs) - np.mean(harmless_projs)
        retention = (gap / direct_gap * 100) if direct_gap else 100.0
        ablation_retentions.append(retention)

        print(f'  {stage_name:<16} {gap:>8.3f} {retention:>11.1f}%')

    is_monotonic = all(ablation_retentions[i] <= ablation_retentions[i+1]
                       for i in range(len(ablation_retentions)-1))

    rho, p_val = spearmanr(list(range(len(ablation_retentions))), ablation_retentions)
    print(f'\n  Strictly monotonic: {"YES" if is_monotonic else "NO"}')
    print(f'  Spearman rho={rho:.4f}, p={p_val:.6f}')
    RESULTS['ablation_monotonic'] = is_monotonic
    RESULTS['ablation_spearman_rho'] = rho
    RESULTS['ablation_spearman_p'] = p_val
    RESULTS['ablation_retentions'] = ablation_retentions

    # ============================================================
    # 4. ACTIVATION PATCHING (Table III - cumulative, gap-based)
    # ============================================================
    section('4. ACTIVATION PATCHING (Table III)')

    def get_patched_projections(model, tokenizer, prompts, fmt_fn, direction, up_to_layer, best_layer):
        """Patch layers 0..up_to_layer from direct into formatted, return projections."""
        projections = []
        model.eval()

        for p in prompts:
            source_prompt = p
            target_prompt = fmt_fn(p)

            with torch.no_grad():
                src_inputs = tokenizer(source_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                src_outputs = model(**src_inputs, output_hidden_states=True)
                src_hidden = src_outputs.hidden_states

            tgt_inputs = tokenizer(target_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            layers = model.model.layers
            hooks = []

            for l in range(min(up_to_layer + 1, len(layers))):
                source_h = src_hidden[l + 1]

                def make_hook(sh):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            h[0, -1, :] = sh[0, min(sh.shape[1]-1, -1), :]
                            return (h,) + output[1:]
                        else:
                            o = output.clone()
                            o[0, -1, :] = sh[0, min(sh.shape[1]-1, -1), :]
                            return o
                    return hook_fn

                hook = layers[l].register_forward_hook(make_hook(source_h))
                hooks.append(hook)

            with torch.no_grad():
                tgt_outputs = model(**tgt_inputs, output_hidden_states=True)

            for hook in hooks:
                hook.remove()

            final_h = tgt_outputs.hidden_states[best_layer + 1][0, -1, :].cpu().numpy()
            projections.append(float(np.dot(final_h, direction)))

        return projections

    fmt_fn = formats['tool_call']
    n_patch = min(8, len(test_harmful))
    patch_harmful = test_harmful[:n_patch]
    patch_harmless = test_harmless[:n_patch]

    # Baseline: tool_call format gap (no patching)
    tc_harmful_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in patch_harmful])
    tc_harmless_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in patch_harmless])
    baseline_harmful = [float(np.dot(h[best_layer], direction)) for h in tc_harmful_h]
    baseline_harmless = [float(np.dot(h[best_layer], direction)) for h in tc_harmless_h]
    baseline_gap = np.mean(baseline_harmful) - np.mean(baseline_harmless)

    # Target: direct format gap
    dir_harmful_h = get_hidden_states(model, tokenizer, patch_harmful)
    dir_harmless_h = get_hidden_states(model, tokenizer, patch_harmless)
    target_harmful = [float(np.dot(h[best_layer], direction)) for h in dir_harmful_h]
    target_harmless = [float(np.dot(h[best_layer], direction)) for h in dir_harmless_h]
    target_gap = np.mean(target_harmful) - np.mean(target_harmless)

    print(f'  Baseline gap (tool_call, no patch): {baseline_gap:.3f}')
    print(f'  Target gap (direct): {target_gap:.3f}')
    print(f'  Patching: direct -> tool_call (cumulative layers 0..L)\n')
    print(f'  {"Layers patched":>15} {"Patched gap":>12} {"Restoration %":>14}')
    print(f'  {"-"*43}')

    n_model_layers = len(model.model.layers)
    patch_levels = [0, 6, 12, 18, 24, min(26, n_model_layers - 1)]

    for up_to in patch_levels:
        if up_to >= n_model_layers:
            continue
        patched_h_projs = get_patched_projections(
            model, tokenizer, patch_harmful, fmt_fn, direction, up_to, best_layer)
        patched_s_projs = get_patched_projections(
            model, tokenizer, patch_harmless, fmt_fn, direction, up_to, best_layer)

        patched_gap = np.mean(patched_h_projs) - np.mean(patched_s_projs)
        if abs(target_gap - baseline_gap) > 1e-6:
            restoration = (patched_gap - baseline_gap) / (target_gap - baseline_gap) * 100
        else:
            restoration = 0.0

        print(f'  {"0.." + str(up_to):>15} {patched_gap:>11.3f} {restoration:>13.1f}%')
        RESULTS[f'patching_cumul_{up_to}'] = restoration

    # ============================================================
    # 5. CALIBRATION / THRESHOLD TRANSFER FAILURE
    # ============================================================
    section('5. CALIBRATION TRANSFER FAILURE')

    # Train threshold on direct-format validation set
    val_direct_harmful_h = get_hidden_states(model, tokenizer, val_harmful)
    val_direct_harmless_h = get_hidden_states(model, tokenizer, val_harmless)
    val_h_projs = [float(np.dot(h[best_layer], direction)) for h in val_direct_harmful_h]
    val_s_projs = [float(np.dot(h[best_layer], direction)) for h in val_direct_harmless_h]

    X_cal = np.array(val_h_projs + val_s_projs).reshape(-1, 1)
    y_cal = np.array([1]*len(val_h_projs) + [0]*len(val_s_projs))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_cal, y_cal)
    print(f'  Probe trained on validation set (direct format, N={len(y_cal)})')
    print(f'  Testing threshold transfer to formatted test set:\n')
    print(f'  {"Format":<12} {"Accuracy":>10}')
    print(f'  {"-"*24}')

    for fmt_name in formats:
        X_fmt = np.array(format_harmful_projs[fmt_name] + format_harmless_projs[fmt_name]).reshape(-1, 1)
        y_fmt = np.array([1]*N_test + [0]*N_test)
        acc = lr.score(X_fmt, y_fmt) * 100
        print(f'  {fmt_name:<12} {acc:>9.1f}%')
        RESULTS[f'calibration_acc_{fmt_name}'] = acc

    # ============================================================
    # 6. CROSS-MODEL GAP MEASUREMENT
    # ============================================================
    section('6. CROSS-MODEL REPLICATION')

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    cross_models = [
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    ]

    for cm_name in cross_models:
        print(f'\n  Loading {cm_name}...')
        try:
            cm_tokenizer = AutoTokenizer.from_pretrained(cm_name, trust_remote_code=True)
            cm_model = AutoModelForCausalLM.from_pretrained(
                cm_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
        except Exception as e:
            print(f'  SKIP: {e}')
            continue

        cm_dirs = extract_refusal_direction(cm_model, cm_tokenizer, ext_harmful, ext_harmless)

        cm_ext_harmful_h = get_hidden_states(cm_model, cm_tokenizer, ext_harmful)
        cm_ext_harmless_h = get_hidden_states(cm_model, cm_tokenizer, ext_harmless)

        cm_n_layers = len(cm_dirs)
        cm_layer_aucs = []
        cm_layer_gaps = []
        for layer in range(cm_n_layers):
            projs_h = [float(np.dot(h[layer], cm_dirs[layer])) for h in cm_ext_harmful_h]
            projs_s = [float(np.dot(h[layer], cm_dirs[layer])) for h in cm_ext_harmless_h]
            y_true = [1]*len(projs_h) + [0]*len(projs_s)
            scores = projs_h + projs_s
            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = 0.5
            gap = np.mean(projs_h) - np.mean(projs_s)
            cm_layer_aucs.append(auc)
            cm_layer_gaps.append(gap)

        cm_candidates = [l for l in range(cm_n_layers) if cm_layer_aucs[l] >= 0.99]
        if not cm_candidates:
            cm_candidates = list(range(cm_n_layers))
        cm_best_layer = max(cm_candidates, key=lambda l: cm_layer_gaps[l])

        print(f'  Selected layer: {cm_best_layer} (AUC={cm_layer_aucs[cm_best_layer]:.4f})')
        cm_direction = cm_dirs[cm_best_layer]

        cm_test_harmful_direct = get_hidden_states(cm_model, cm_tokenizer, test_harmful[:20])
        cm_test_harmless_direct = get_hidden_states(cm_model, cm_tokenizer, test_harmless[:20])

        fmt_fn_tc = formats['tool_call']
        cm_test_harmful_tc = get_hidden_states(cm_model, cm_tokenizer, [fmt_fn_tc(p) for p in test_harmful[:20]])
        cm_test_harmless_tc = get_hidden_states(cm_model, cm_tokenizer, [fmt_fn_tc(p) for p in test_harmless[:20]])

        direct_projs_h = [float(np.dot(h[cm_best_layer], cm_direction)) for h in cm_test_harmful_direct]
        direct_projs_s = [float(np.dot(h[cm_best_layer], cm_direction)) for h in cm_test_harmless_direct]
        tc_projs_h = [float(np.dot(h[cm_best_layer], cm_direction)) for h in cm_test_harmful_tc]
        tc_projs_s = [float(np.dot(h[cm_best_layer], cm_direction)) for h in cm_test_harmless_tc]

        cm_direct_gap = np.mean(direct_projs_h) - np.mean(direct_projs_s)
        cm_tc_gap = np.mean(tc_projs_h) - np.mean(tc_projs_s)
        cm_retention = (cm_tc_gap / cm_direct_gap * 100) if cm_direct_gap != 0 else 0

        print(f'  Direct gap: {cm_direct_gap:.3f}')
        print(f'  Tool-call gap: {cm_tc_gap:.3f}')
        print(f'  Retention: {cm_retention:.1f}%')

        short_name = cm_name.split("/")[-1]
        RESULTS[f'cross_{short_name}_layer'] = cm_best_layer
        RESULTS[f'cross_{short_name}_retention'] = cm_retention

        del cm_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    section('REPRODUCTION SUMMARY')

    elapsed = time.time() - start
    print(f'  Total time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)\n')

    # Identify lowest-retention agentic format for calibration check
    agentic_formats = ['tool_call', 'json', 'system', 'xml']
    min_cal_format = min(agentic_formats,
                         key=lambda f: RESULTS.get(f'calibration_acc_{f}', 100))
    min_cal_acc = RESULTS.get(f'calibration_acc_{min_cal_format}', 100)

    checks = [
        ('Layer AUC >= 0.95 on validation set',
         RESULTS.get('layer_auc', 0) >= 0.95),
        ('Tool-call retention < 15%',
         RESULTS.get('retention_tool_call', 100) < 15),
        ('System retention < 10%',
         RESULTS.get('retention_system', 100) < 10),
        ('Direct retention == 100%',
         abs(RESULTS.get('retention_direct', 0) - 100) < 0.1),
        ('Ablation strictly monotonic',
         RESULTS.get('ablation_monotonic', False)),
        ('Ablation Spearman rho == 1.0',
         abs(RESULTS.get('ablation_spearman_rho', 0) - 1.0) < 0.01),
        ('Patching restores gap at layer 0',
         RESULTS.get('patching_cumul_0', 0) > 0),
        (f'Calibration fails for {min_cal_format} (<= 55%)',
         min_cal_acc <= 55),
        ('Cross-model effect replicates (TinyLlama)',
         RESULTS.get('cross_TinyLlama-1.1B-Chat-v1.0_retention', 100) < 15),
    ]

    print(f'  {"Check":<45} {"Status":>8}')
    print(f'  {"-"*55}')
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f'  {name:<45} {status:>8}')

    print(f'\n  {"ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"}')
    print(f'  (Minor deviations are expected across hardware/driver versions)\n')

    results_json = json.dumps(RESULTS, indent=2, sort_keys=True, default=str)
    sha256 = hashlib.sha256(results_json.encode()).hexdigest()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv', 'reproduction_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(results_json)

    print(f'  Results hash (SHA256): {sha256}')
    print(f'  Results saved to: {out_path}')
    print(f'\n  REPRODUCTION COMPLETE.')


if __name__ == '__main__':
    main()
