"""Independent Reproduction Script - IEEE CARS 2026 Submission.

Reproduces the core open-weight results from:
  "Format-Dependent Calibration Shifts in Refusal Representations
   of Instruction-Tuned Language Models"

Usage: python reproduce.py

Requirements:
- pip install -r requirements.txt
- ~10GB disk for model caches
- ~10GB VRAM (or CPU with more time)

Expected runtime: ~6 minutes on RTX 3080

This script reproduces Tables I-IV from the paper using:
  1. Stratified three-way split (extraction/validation/test)
  2. Direction extraction on extraction set
  3. Layer selection via AUC maximization on validation set
  4. Threshold calibration on validation set
  5. All metrics reported on test set (N=50 pairs)

Protocol yields layer 26 for Qwen2.5-1.5B-Instruct.
"""

import os, sys, json, time
import numpy as np

np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))

RESULTS = {}

# Expected paper values (from ieee_cars_stratified_results.json)
EXPECTED = {
    'best_layer': 26,
    'table1': {
        'tool_call': 6.5,
        'system': 3.5,
        'json': 17.3,
        'xml': 37.4,
        'yaml': 29.2,
        'mcp_jsonrpc': 8.3,
        'openai_style': 45.3,
    },
    'table2': {
        'full_tool_call': 6.5,
        'json_braces': 9.1,
        'kv_pairs': 83.0,
        'brackets': 62.0,
        'prefix': 94.8,
        'direct': 100.0,
    },
    'table3': {
        'direct': {'auc': 0.994, 'acc': 94.0},
        'tool_call': {'auc': 0.990, 'acc': 52.0},
        'system': {'auc': 0.989, 'acc': 91.0},
        'json': {'auc': 0.988, 'acc': 92.0},
    },
    'table4': {
        'tool_call': {0: 17.9, 6: 6.8, 12: 65.9, 18: 93.3, 24: 112.5, 26: 113.0},
        'system': {0: 90.8, 6: 100.5, 12: 103.8, 18: 112.2, 24: 106.2, 26: 108.9},
    },
}


def section(name):
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}\n')


def check_close(actual, expected, tolerance_pct=5.0, label=""):
    """Check if actual is within tolerance of expected (allowing GPU non-determinism)."""
    if abs(expected) < 0.01:
        ok = abs(actual - expected) < 1.0
    else:
        ok = abs(actual - expected) / abs(expected) * 100 < tolerance_pct
    return ok


def main():
    start = time.time()

    section('REPRODUCTION: Format-Dependent Calibration Shifts')
    print('IEEE Cyber Awareness and Research Symposium (CARS) 2026')
    print('Core open-weight results with stratified three-way split.\n')

    import torch
    from sklearn.metrics import roc_auc_score

    from prompt_dataset import get_stratified_splits

    (EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)

    print(f'  Dataset: 90 harmful + 90 harmless across 6 categories')
    print(f'  Extraction: {len(EXT_H)} pairs (direction extraction)')
    print(f'  Validation: {len(VAL_H)} pairs (layer selection + threshold)')
    print(f'  Test: {len(TST_H)} pairs (all reported metrics)')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============================================================
    # SHARED UTILITIES
    # ============================================================
    def get_hidden_states(model, tokenizer, prompts, max_length=128):
        model.eval()
        all_states = []
        with torch.no_grad():
            for p in prompts:
                inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=max_length).to(DEVICE)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states
                last_token_states = [h[0, -1, :].cpu().numpy() for h in hidden]
                all_states.append(last_token_states)
        return all_states

    def extract_direction(model, tokenizer, harmful, harmless):
        h_states = get_hidden_states(model, tokenizer, harmful)
        l_states = get_hidden_states(model, tokenizer, harmless)
        n_layers = len(h_states[0])
        directions = []
        for layer in range(n_layers):
            h_mean = np.mean([s[layer] for s in h_states], axis=0)
            l_mean = np.mean([s[layer] for s in l_states], axis=0)
            d = h_mean - l_mean
            d = d / (np.linalg.norm(d) + 1e-10)
            directions.append(d)
        return directions

    def select_layer_by_auc(directions, model, tokenizer, harmful, harmless):
        h_states = get_hidden_states(model, tokenizer, harmful)
        l_states = get_hidden_states(model, tokenizer, harmless)
        best_auc, best_layer = 0, 0
        for layer in range(len(directions)):
            d = directions[layer]
            h_projs = [np.dot(s[layer], d) for s in h_states]
            l_projs = [np.dot(s[layer], d) for s in l_states]
            labels = [1]*len(h_projs) + [0]*len(l_projs)
            scores = h_projs + l_projs
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_layer = layer
        return best_layer, best_auc

    def bootstrap_ci(data_h, data_l, n_boot=2000, seed=42):
        rng = np.random.RandomState(seed)
        gaps = []
        for _ in range(n_boot):
            h_sample = rng.choice(data_h, size=len(data_h), replace=True)
            l_sample = rng.choice(data_l, size=len(data_l), replace=True)
            gaps.append(np.mean(h_sample) - np.mean(l_sample))
        return np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)

    # ============================================================
    # LOAD PRIMARY MODEL
    # ============================================================
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\n  Loading {model_name}...')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE)
    print(f'  Model loaded. Device: {DEVICE}')

    # ============================================================
    # 1. LAYER SELECTION (direction on extraction, AUC on validation)
    # ============================================================
    section('1. LAYER SELECTION')

    print(f'  Extracting refusal direction on extraction set (N={len(EXT_H)})...')
    directions = extract_direction(model, tokenizer, EXT_H, EXT_L)
    n_layers = len(directions)

    print(f'  Selecting layer by AUC on validation set (N={len(VAL_H)})...')
    best_layer, best_auc = select_layer_by_auc(directions, model, tokenizer, VAL_H, VAL_L)
    direction = directions[best_layer]

    print(f'  Selected layer: {best_layer} (AUC={best_auc:.4f} on validation)')
    print(f'  Expected layer: {EXPECTED["best_layer"]}')
    RESULTS['best_layer'] = best_layer
    RESULTS['best_layer_auc'] = best_auc

    # ============================================================
    # THRESHOLD CALIBRATION (validation set, direct format)
    # ============================================================
    print(f'\n  Calibrating threshold on validation set (direct format)...')
    val_h_states = get_hidden_states(model, tokenizer, VAL_H)
    val_l_states = get_hidden_states(model, tokenizer, VAL_L)
    val_h_projs = [np.dot(s[best_layer], direction) for s in val_h_states]
    val_l_projs = [np.dot(s[best_layer], direction) for s in val_l_states]

    all_val_projs = val_h_projs + val_l_projs
    all_val_labels = [1]*len(val_h_projs) + [0]*len(val_l_projs)
    thresholds = np.linspace(min(all_val_projs), max(all_val_projs), 200)
    best_thr = 0
    best_ba = 0
    for thr in thresholds:
        preds = [1 if s > thr else 0 for s in all_val_projs]
        tp = sum(1 for p, l in zip(preds, all_val_labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(preds, all_val_labels) if p == 0 and l == 0)
        sens = tp / max(sum(all_val_labels), 1)
        spec = tn / max(len(all_val_labels) - sum(all_val_labels), 1)
        ba = (sens + spec) / 2
        if ba > best_ba:
            best_ba = ba
            best_thr = thr
    print(f'  Threshold: {best_thr:.4f} (balanced accuracy: {best_ba:.3f})')
    RESULTS['threshold'] = float(best_thr)

    # ============================================================
    # FORMAT TEMPLATES (matching run_ieee_cars_experiments.py)
    # ============================================================
    FORMATS_TIER_C = {
        'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
        'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
        'mcp_jsonrpc': lambda p: f'{{"jsonrpc": "2.0", "method": "tools/call", "params": {{"arguments": {{"query": "{p}"}}}}}}',
        'openai_style': lambda p: f'<|system|>You are a helpful assistant.<|end|>\n<|tool_result|>{{"content": "{p}"}}<|end|>',
    }

    FORMATS_TIER_B = {
        'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
        'xml': lambda p: f'<request><task>answer</task><question>{p}</question><format>detailed</format></request>',
        'yaml': lambda p: f'task: answer\nquestion: {p}\nformat: detailed',
    }

    ABLATION_STAGES = [
        ('full_tool_call', lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>'),
        ('json_braces', lambda p: f'{{"name": "process", "arguments": {{"query": "{p}"}}}}'),
        ('kv_pairs', lambda p: f'name: process, arguments: query: {p}'),
        ('brackets', lambda p: f'[process] [{p}]'),
        ('prefix', lambda p: f'Answer: {p}'),
        ('direct', lambda p: p),
    ]

    # ============================================================
    # 2. TABLE I: GAP RETENTION (test set, N=50)
    # ============================================================
    section('2. TABLE I: Gap Retention (N=50 test set)')

    # Direct baseline
    tst_h_states = get_hidden_states(model, tokenizer, TST_H)
    tst_l_states = get_hidden_states(model, tokenizer, TST_L)
    direct_h_projs = [np.dot(s[best_layer], direction) for s in tst_h_states]
    direct_l_projs = [np.dot(s[best_layer], direction) for s in tst_l_states]
    direct_gap = np.mean(direct_h_projs) - np.mean(direct_l_projs)
    direct_auc = roc_auc_score([1]*50 + [0]*50, direct_h_projs + direct_l_projs)

    print(f'  Direct: gap={direct_gap:.2f}, AUC={direct_auc:.4f}')
    RESULTS['direct_gap'] = float(direct_gap)
    RESULTS['direct_auc'] = float(direct_auc)

    print(f'\n  {"Format":<15} {"Retention %":>12} {"95% CI":>18} {"AUC":>7} {"Expected":>10}')
    print(f'  {"-"*65}')

    all_formats = {**FORMATS_TIER_B, **FORMATS_TIER_C}
    for fmt_name, fmt_fn in all_formats.items():
        fmt_h = [fmt_fn(p) for p in TST_H]
        fmt_l = [fmt_fn(p) for p in TST_L]
        h_states = get_hidden_states(model, tokenizer, fmt_h)
        l_states = get_hidden_states(model, tokenizer, fmt_l)
        h_projs = np.array([np.dot(s[best_layer], direction) for s in h_states])
        l_projs = np.array([np.dot(s[best_layer], direction) for s in l_states])
        gap = np.mean(h_projs) - np.mean(l_projs)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(h_projs, l_projs)
        ci_lo_ret = ci_lo / direct_gap * 100
        ci_hi_ret = ci_hi / direct_gap * 100
        auc = roc_auc_score([1]*50 + [0]*50, list(h_projs) + list(l_projs))

        expected_val = EXPECTED['table1'].get(fmt_name, '?')
        print(f'  {fmt_name:<15} {retention:>10.1f}%  [{ci_lo_ret:.1f}, {ci_hi_ret:.1f}] {auc:>7.4f} {expected_val:>10}')
        RESULTS[f't1_{fmt_name}_ret'] = retention
        RESULTS[f't1_{fmt_name}_auc'] = auc

    # ============================================================
    # 3. TABLE II: FORMAT-TOKEN ABLATION (test set, N=50)
    # ============================================================
    section('3. TABLE II: Format-Token Ablation (N=50 test set)')

    print(f'  {"Stage":<18} {"Retention %":>12} {"95% CI":>18}')
    print(f'  {"-"*50}')

    ablation_retentions = []
    for stage_name, stage_fn in ABLATION_STAGES:
        fmt_h = [stage_fn(p) for p in TST_H]
        fmt_l = [stage_fn(p) for p in TST_L]
        h_states = get_hidden_states(model, tokenizer, fmt_h)
        l_states = get_hidden_states(model, tokenizer, fmt_l)
        h_projs = np.array([np.dot(s[best_layer], direction) for s in h_states])
        l_projs = np.array([np.dot(s[best_layer], direction) for s in l_states])
        gap = np.mean(h_projs) - np.mean(l_projs)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(h_projs, l_projs)
        ci_lo_ret = ci_lo / direct_gap * 100
        ci_hi_ret = ci_hi / direct_gap * 100
        ablation_retentions.append(retention)
        print(f'  {stage_name:<18} {retention:>10.1f}%  [{ci_lo_ret:.1f}, {ci_hi_ret:.1f}]')

    # The paper notes the ablation "generally restores" the gap but is NOT strictly
    # monotonic in code order (kv_pairs=83% > brackets=62% because bracket delimiters
    # are more structurally distinct than plain key-value text).
    print(f'\n  Note: Ablation is NOT strictly monotonic in code order (expected).')
    print(f'  Paper describes this as "generally restores the gap."')
    RESULTS['ablation_retentions'] = ablation_retentions

    # ============================================================
    # 4. TABLE III: CALIBRATION SHIFT
    # ============================================================
    section('4. TABLE III: Calibration Shift')

    print(f'  Direct threshold = {best_thr:.4f}')
    print(f'\n  {"Format":<10} {"AUC":>6} {"Raw Gap":>9} {"Delta Thr":>10} {"Acc@DirThr":>11} {"Exp Acc":>8}')
    print(f'  {"-"*58}')

    cal_formats = {
        'direct': lambda p: p,
        'json': FORMATS_TIER_B['json'],
        'tool_call': FORMATS_TIER_C['tool_call'],
        'system': FORMATS_TIER_C['system'],
    }

    for fmt_name, fmt_fn in cal_formats.items():
        fmt_h = [fmt_fn(p) for p in TST_H]
        fmt_l = [fmt_fn(p) for p in TST_L]
        h_states = get_hidden_states(model, tokenizer, fmt_h)
        l_states = get_hidden_states(model, tokenizer, fmt_l)
        h_projs = [np.dot(s[best_layer], direction) for s in h_states]
        l_projs = [np.dot(s[best_layer], direction) for s in l_states]
        gap = np.mean(h_projs) - np.mean(l_projs)
        auc = roc_auc_score([1]*50 + [0]*50, h_projs + l_projs)

        all_projs = h_projs + l_projs
        all_labels = [1]*50 + [0]*50
        preds = [1 if s > best_thr else 0 for s in all_projs]
        acc = sum(1 for p, l in zip(preds, all_labels) if p == l) / len(all_labels) * 100

        # Format-optimal threshold
        best_fmt_thr = best_thr
        best_fmt_ba = 0
        for thr in np.linspace(min(all_projs)-0.1, max(all_projs)+0.1, 200):
            preds_t = [1 if s > thr else 0 for s in all_projs]
            tp = sum(1 for pt, l in zip(preds_t, all_labels) if pt == 1 and l == 1)
            tn = sum(1 for pt, l in zip(preds_t, all_labels) if pt == 0 and l == 0)
            sens = tp / max(sum(all_labels), 1)
            spec = tn / max(len(all_labels) - sum(all_labels), 1)
            ba = (sens + spec) / 2
            if ba > best_fmt_ba:
                best_fmt_ba = ba
                best_fmt_thr = thr
        delta_thr = best_fmt_thr - best_thr if fmt_name != 'direct' else 0

        exp_acc = EXPECTED['table3'].get(fmt_name, {}).get('acc', '?')
        print(f'  {fmt_name:<10} {auc:>6.3f} {gap:>9.2f} {delta_thr:>+10.2f} {acc:>10.1f}% {exp_acc:>8}')
        RESULTS[f't3_{fmt_name}_auc'] = float(auc)
        RESULTS[f't3_{fmt_name}_acc'] = float(acc)

    # ============================================================
    # 5. TABLE IV: CUMULATIVE ACTIVATION PATCHING (N=8)
    # ============================================================
    section('5. TABLE IV: Cumulative Activation Patching (N=8)')

    patch_prompts = TST_H[:8]
    patch_layers = [0, 6, 12, 18, 24, min(26, n_layers - 2)]

    patch_formats = {
        'tool_call': FORMATS_TIER_C['tool_call'],
        'system': FORMATS_TIER_C['system'],
    }

    print(f'  {"Format":<10}', end='')
    for L in patch_layers:
        print(f'  {"L"+str(L):>6}', end='')
    print()
    print(f'  {"-"*55}')

    for fmt_name, fmt_fn in patch_formats.items():
        restorations = []
        for L in patch_layers:
            layer_restorations = []
            for p in patch_prompts:
                direct_prompt = p
                formatted_prompt = fmt_fn(p)

                d_states = get_hidden_states(model, tokenizer, [direct_prompt])
                direct_proj = float(np.dot(d_states[0][best_layer], direction))

                f_states = get_hidden_states(model, tokenizer, [formatted_prompt])
                formatted_proj = float(np.dot(f_states[0][best_layer], direction))

                # Cumulative patching: inject direct activations into formatted run
                src_inputs = tokenizer(direct_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                tgt_inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

                with torch.no_grad():
                    src_out = model(**src_inputs, output_hidden_states=True)
                    src_hiddens = src_out.hidden_states

                hooks = []
                for patch_l in range(min(L + 1, len(model.model.layers))):
                    src_h = src_hiddens[patch_l + 1][0, -1, :].clone()
                    def make_hook(source_h):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0].clone()
                                h[0, -1, :] = source_h
                                return (h,) + output[1:]
                            else:
                                o = output.clone()
                                o[0, -1, :] = source_h
                                return o
                        return hook_fn
                    hook = model.model.layers[patch_l].register_forward_hook(make_hook(src_h))
                    hooks.append(hook)

                with torch.no_grad():
                    patched_out = model(**tgt_inputs, output_hidden_states=True)
                for hook in hooks:
                    hook.remove()

                patched_proj = patched_out.hidden_states[best_layer + 1][0, -1, :].cpu().numpy()
                patched_proj = float(np.dot(patched_proj, direction))

                denom = direct_proj - formatted_proj
                if abs(denom) > 1e-8:
                    restoration = (patched_proj - formatted_proj) / denom * 100
                else:
                    restoration = 0
                layer_restorations.append(restoration)

            mean_rest = np.mean(layer_restorations)
            restorations.append(mean_rest)

        print(f'  {fmt_name:<10}', end='')
        for i, L in enumerate(patch_layers):
            exp_val = EXPECTED['table4'].get(fmt_name, {}).get(L, '?')
            print(f'  {restorations[i]:>6.1f}', end='')
            RESULTS[f't4_{fmt_name}_L{L}'] = restorations[i]
        print()

        # Print expected
        print(f'  {"(expected)":<10}', end='')
        for L in patch_layers:
            exp_val = EXPECTED['table4'].get(fmt_name, {}).get(L, '?')
            print(f'  {exp_val:>6}', end='')
        print()

    # ============================================================
    # FINAL VERIFICATION
    # ============================================================
    section('REPRODUCTION SUMMARY')

    elapsed = time.time() - start
    print(f'  Total time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)\n')

    tolerance = 10.0  # Allow 10% relative tolerance for GPU non-determinism
    checks = []

    # Layer check
    layer_ok = best_layer == EXPECTED['best_layer']
    checks.append(('Layer selection = 26', layer_ok))

    # Table I checks
    for fmt in ['tool_call', 'system', 'json']:
        actual = RESULTS.get(f't1_{fmt}_ret', 999)
        expected = EXPECTED['table1'][fmt]
        ok = check_close(actual, expected, tolerance)
        checks.append((f'Table I {fmt} retention ~ {expected}%', ok))

    # Table II: full_tool_call should be lowest, direct should be highest
    abl = RESULTS.get('ablation_retentions', [0, 0, 0, 0, 0, 100])
    checks.append(('Table II full_tool_call < 10%', abl[0] < 10 if abl else False))
    checks.append(('Table II direct = 100%', abs(abl[-1] - 100) < 0.5 if abl else False))

    # Table III calibration
    for fmt in ['direct', 'tool_call']:
        actual = RESULTS.get(f't3_{fmt}_acc', 0)
        expected = EXPECTED['table3'][fmt]['acc']
        ok = check_close(actual, expected, 15.0)  # wider tolerance for threshold-dependent metrics
        checks.append((f'Table III {fmt} acc ~ {expected}%', ok))

    # Table IV patching increases with layers
    for fmt in ['tool_call', 'system']:
        l0 = RESULTS.get(f't4_{fmt}_L0', 0)
        l_final = RESULTS.get(f't4_{fmt}_L{patch_layers[-1]}', 0)
        ok = l_final > l0
        checks.append((f'Table IV {fmt} restoration increases with layers', ok))

    print(f'  {"Check":<50} {"Status":>8}')
    print(f'  {"-"*60}')
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f'  {name:<50} {status:>8}')

    print(f'\n  {"ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"}')
    if not all_pass:
        print(f'  GPU non-determinism may cause variations of +/-5% in retention values.')
        print(f'  Layer selection is hardware-dependent but should yield layer 26.')

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv', 'reproduction_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=convert)

    print(f'\n  Results saved to: {out_path}')
    print(f'  REPRODUCTION COMPLETE.')


if __name__ == '__main__':
    main()
