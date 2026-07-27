"""IEEE CARS 2026 - Full experiment pipeline with stratified splits.

Produces all open-weight table values for the paper:
- Table I: Gap retention by format tier
- Table II: Format-token ablation
- Table III: Calibration shift signature
- Table IV: Cumulative activation patching
- Table V: Cross-model replication

Usage: python run_ieee_cars_experiments.py
Hardware: RTX 3080 (10GB), ~2 hours total
"""

import os, sys, json, time
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits, HARMFUL_CATEGORIES, HARMLESS_CATEGORIES

from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)

# Get stratified splits
(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)


def get_hidden_states(model, tokenizer, prompts, max_length=128):
    """Extract last-token hidden states at all layers."""
    model.eval()
    all_states = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=max_length).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states  # tuple of (n_layers+1) tensors
            last_token_states = [h[0, -1, :].cpu().numpy() for h in hidden]
            all_states.append(last_token_states)
    return all_states


def extract_direction(model, tokenizer, harmful_prompts, harmless_prompts):
    """Extract refusal direction at each layer using difference-in-means."""
    h_states = get_hidden_states(model, tokenizer, harmful_prompts)
    l_states = get_hidden_states(model, tokenizer, harmless_prompts)
    n_layers = len(h_states[0])
    directions = []
    for layer in range(n_layers):
        h_mean = np.mean([s[layer] for s in h_states], axis=0)
        l_mean = np.mean([s[layer] for s in l_states], axis=0)
        d = h_mean - l_mean
        d = d / (np.linalg.norm(d) + 1e-10)
        directions.append(d)
    return directions


def select_layer_by_auc(directions, model, tokenizer, harmful_prompts, harmless_prompts):
    """Select best layer by AUC on given prompts."""
    h_states = get_hidden_states(model, tokenizer, harmful_prompts)
    l_states = get_hidden_states(model, tokenizer, harmless_prompts)
    n_layers = len(directions)
    best_auc = 0
    best_layer = 0
    for layer in range(n_layers):
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


def compute_gap_and_auc(direction, layer, model, tokenizer, harmful_prompts, harmless_prompts):
    """Compute gap retention, AUC, and projections for given prompts."""
    h_states = get_hidden_states(model, tokenizer, harmful_prompts)
    l_states = get_hidden_states(model, tokenizer, harmless_prompts)
    h_projs = [np.dot(s[layer], direction) for s in h_states]
    l_projs = [np.dot(s[layer], direction) for s in l_states]
    gap = np.mean(h_projs) - np.mean(l_projs)
    labels = [1]*len(h_projs) + [0]*len(l_projs)
    scores = h_projs + l_projs
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    return gap, auc, h_projs, l_projs


def bootstrap_ci(data_h, data_l, n_boot=2000, seed=42):
    """Bootstrap 95% CI on gap."""
    rng = np.random.RandomState(seed)
    gaps = []
    for _ in range(n_boot):
        h_sample = rng.choice(data_h, size=len(data_h), replace=True)
        l_sample = rng.choice(data_l, size=len(data_l), replace=True)
        gaps.append(np.mean(h_sample) - np.mean(l_sample))
    return np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)


# ============================================================
# FORMAT TEMPLATES
# ============================================================

FORMATS_TIER_C = {
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'openai_style': lambda p: f'<|system|>You are a helpful assistant.<|end|>\n<|tool_result|>{{"content": "{p}"}}<|end|>',
    'mcp_jsonrpc': lambda p: f'{{"jsonrpc": "2.0", "method": "tools/call", "params": {{"arguments": {{"query": "{p}"}}}}}}',
}

FORMATS_TIER_B = {
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'xml': lambda p: f'<request><task>answer</task><question>{p}</question><format>detailed</format></request>',
    'yaml': lambda p: f'task: answer\nquestion: {p}\nformat: detailed',
}

ABLATION_STAGES = {
    'full_tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'json_braces': lambda p: f'{{"name": "process", "arguments": {{"query": "{p}"}}}}',
    'kv_pairs': lambda p: f'name: process, arguments: query: {p}',
    'brackets': lambda p: f'[process] [{p}]',
    'prefix': lambda p: f'Answer: {p}',
    'direct': lambda p: p,
}

LENGTH_MATCHED = {
    'natural_filler': lambda p: f'Please answer the following question for me: {p}',
    'tool_format_32': lambda p: f'<tool_call>\n{{"name": "x", "arguments": {{"q": "{p}"}}}}\n</tool_call>',
}

TOKEN_IDENTITY = {
    'real_token': lambda p: f'<tool_call>{p}</tool_call>',
    'arbitrary_delims': lambda p: f'<<<FUNC>>>{p}<<</FUNC>>>',
}

# Standardized cross-model formats (shared across all architectures)
CROSS_MODEL_FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}


def run_primary_model():
    """Run all experiments on Qwen2.5-1.5B-Instruct."""
    print("\n" + "="*70)
    print("  PRIMARY MODEL: Qwen2.5-1.5B-Instruct")
    print("="*70)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE)

    # Step 1: Extract direction on EXTRACTION set
    print('\n--- Direction extraction (N=20 extraction set) ---')
    directions = extract_direction(model, tokenizer, EXT_H, EXT_L)

    # Step 2: Select layer by AUC on VALIDATION set (directions extracted on extraction set)
    best_layer, best_auc = select_layer_by_auc(directions, model, tokenizer, VAL_H, VAL_L)
    print(f'Best layer: {best_layer} (AUC={best_auc:.4f} on validation set)')

    direction = directions[best_layer]

    # Step 3: Validate threshold on VALIDATION set
    print('\n--- Threshold calibration (N=20 validation set) ---')
    val_gap, val_auc, val_h_projs, val_l_projs = compute_gap_and_auc(
        direction, best_layer, model, tokenizer, VAL_H, VAL_L)
    all_val_projs = val_h_projs + val_l_projs
    all_val_labels = [1]*len(val_h_projs) + [0]*len(val_l_projs)
    # Threshold = maximize balanced accuracy
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
    print(f'Direct threshold (validation): {best_thr:.4f}, balanced acc: {best_ba:.3f}')

    # Step 4: TEST SET - Table I (gap retention by format)
    print('\n--- TABLE I: Gap retention (N=50 test set) ---')
    direct_gap, direct_auc, direct_h, direct_l = compute_gap_and_auc(
        direction, best_layer, model, tokenizer, TST_H, TST_L)
    print(f'  Direct: gap={direct_gap:.4f}, AUC={direct_auc:.4f}')

    results_table1 = {'direct': {'retention': 100.0, 'gap': direct_gap, 'auc': direct_auc}}

    all_formats = {**FORMATS_TIER_B, **FORMATS_TIER_C}
    for fmt_name, fmt_fn in all_formats.items():
        fmt_h = [fmt_fn(p) for p in TST_H]
        fmt_l = [fmt_fn(p) for p in TST_L]
        gap, auc, h_projs, l_projs = compute_gap_and_auc(
            direction, best_layer, model, tokenizer, fmt_h, fmt_l)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(np.array(h_projs), np.array(l_projs))
        ci_lo_ret = (ci_lo / direct_gap * 100)
        ci_hi_ret = (ci_hi / direct_gap * 100)
        print(f'  {fmt_name:<15} ret={retention:.1f}% [{ci_lo_ret:.1f}, {ci_hi_ret:.1f}] AUC={auc:.4f}')
        results_table1[fmt_name] = {
            'retention': retention, 'ci_lo': ci_lo_ret, 'ci_hi': ci_hi_ret,
            'gap': gap, 'auc': auc
        }

    # Step 5: TEST SET - Table II (ablation)
    print('\n--- TABLE II: Ablation (N=50 test set) ---')
    results_table2 = {}
    ablation_retentions = []
    for stage_name, stage_fn in ABLATION_STAGES.items():
        fmt_h = [stage_fn(p) for p in TST_H]
        fmt_l = [stage_fn(p) for p in TST_L]
        gap, auc, h_projs, l_projs = compute_gap_and_auc(
            direction, best_layer, model, tokenizer, fmt_h, fmt_l)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(np.array(h_projs), np.array(l_projs))
        ci_lo_ret = (ci_lo / direct_gap * 100)
        ci_hi_ret = (ci_hi / direct_gap * 100)
        ablation_retentions.append(retention)
        print(f'  {stage_name:<15} ret={retention:.1f}% [{ci_lo_ret:.1f}, {ci_hi_ret:.1f}]')
        results_table2[stage_name] = {'retention': retention, 'ci_lo': ci_lo_ret, 'ci_hi': ci_hi_ret}

    # Length-matched controls
    print('  --- Length-matched ---')
    for ctrl_name, ctrl_fn in LENGTH_MATCHED.items():
        fmt_h = [ctrl_fn(p) for p in TST_H]
        fmt_l = [ctrl_fn(p) for p in TST_L]
        gap, auc, h_projs, l_projs = compute_gap_and_auc(
            direction, best_layer, model, tokenizer, fmt_h, fmt_l)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(np.array(h_projs), np.array(l_projs))
        print(f'  {ctrl_name:<20} ret={retention:.1f}% [{ci_lo/direct_gap*100:.1f}, {ci_hi/direct_gap*100:.1f}]')
        results_table2[ctrl_name] = {'retention': retention, 'ci_lo': ci_lo/direct_gap*100, 'ci_hi': ci_hi/direct_gap*100}

    # Token-identity controls
    print('  --- Token identity ---')
    for tok_name, tok_fn in TOKEN_IDENTITY.items():
        fmt_h = [tok_fn(p) for p in TST_H]
        fmt_l = [tok_fn(p) for p in TST_L]
        gap, auc, h_projs, l_projs = compute_gap_and_auc(
            direction, best_layer, model, tokenizer, fmt_h, fmt_l)
        retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
        ci_lo, ci_hi = bootstrap_ci(np.array(h_projs), np.array(l_projs))
        print(f'  {tok_name:<20} ret={retention:.1f}% [{ci_lo/direct_gap*100:.1f}, {ci_hi/direct_gap*100:.1f}]')
        results_table2[tok_name] = {'retention': retention, 'ci_lo': ci_lo/direct_gap*100, 'ci_hi': ci_hi/direct_gap*100}

    # Spearman on ablation
    rho, p_val = stats.spearmanr(range(len(ablation_retentions)), ablation_retentions)
    print(f'\n  Ablation Spearman rho={rho:.2f}, p={p_val:.4f}')

    # Step 6: Table III (calibration)
    print('\n--- TABLE III: Calibration shift ---')
    results_table3 = {}
    cal_formats = {'direct': lambda p: p, 'json': FORMATS_TIER_B['json'],
                   'tool_call': FORMATS_TIER_C['tool_call'], 'system': FORMATS_TIER_C['system']}
    for fmt_name, fmt_fn in cal_formats.items():
        fmt_h = [fmt_fn(p) for p in TST_H]
        fmt_l = [fmt_fn(p) for p in TST_L]
        gap, auc, h_projs, l_projs = compute_gap_and_auc(
            direction, best_layer, model, tokenizer, fmt_h, fmt_l)

        all_projs = h_projs + l_projs
        all_labels = [1]*len(h_projs) + [0]*len(l_projs)
        # Accuracy at direct threshold
        preds = [1 if s > best_thr else 0 for s in all_projs]
        acc = sum(1 for p, l in zip(preds, all_labels) if p == l) / len(all_labels) * 100

        # Format-specific optimal threshold
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
        print(f'  {fmt_name:<10} AUC={auc:.3f} raw_delta={gap:.3f} delta_thr={delta_thr:.3f} acc@dir_thr={acc:.1f}%')
        results_table3[fmt_name] = {'auc': auc, 'raw_delta': gap, 'delta_thr': delta_thr, 'acc_at_dir_thr': acc}

    # Step 7: Table IV (cumulative patching, N=8)
    print('\n--- TABLE IV: Cumulative activation patching (N=8) ---')
    results_table4 = {}
    patch_prompts = TST_H[:8]
    patch_layers = [0, 6, 12, 18, 24, 26]

    for fmt_name in ['tool_call', 'system']:
        fmt_fn = FORMATS_TIER_C[fmt_name] if fmt_name != 'system' else FORMATS_TIER_C['system']
        restorations = []
        for L in patch_layers:
            layer_restorations = []
            for p in patch_prompts:
                direct_prompt = p
                formatted_prompt = fmt_fn(p)

                # Get direct projection
                d_states = get_hidden_states(model, tokenizer, [direct_prompt])
                direct_proj = np.dot(d_states[0][best_layer], direction)

                # Get formatted projection
                f_states = get_hidden_states(model, tokenizer, [formatted_prompt])
                formatted_proj = np.dot(f_states[0][best_layer], direction)

                # Cumulative patching: patch layers 0..L
                src_inputs = tokenizer(direct_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                tgt_inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

                with torch.no_grad():
                    src_out = model(**src_inputs, output_hidden_states=True)
                    src_hiddens = src_out.hidden_states

                hooks = []
                for patch_l in range(L + 1):
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
            print(f'  {fmt_name} L0..{L}: {mean_rest:.1f}%')

        results_table4[fmt_name] = dict(zip(patch_layers, restorations))

    # Save results
    all_results = {
        'best_layer': best_layer,
        'best_layer_auc': best_auc,
        'direct_threshold': float(best_thr),
        'direct_gap': float(direct_gap),
        'table1': results_table1,
        'table2': results_table2,
        'table3': results_table3,
        'table4': results_table4,
    }

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return all_results


def run_cross_model(primary_results):
    """Run cross-model replication (Table V)."""
    print("\n" + "="*70)
    print("  CROSS-MODEL REPLICATION")
    print("="*70)

    models = [
        ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B', {}),
        ('Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5-3B', {'device_map': 'auto'}),
        ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'TinyLlama-1.1B', {}),
        ('HuggingFaceTB/SmolLM2-1.7B-Instruct', 'SmolLM2-1.7B', {}),
        ('microsoft/phi-2', 'Phi-2', {}),
    ]

    results_table5 = {}

    for model_id, model_label, extra_kwargs in models:
        print(f'\n  Loading {model_label} ({model_id})...')
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float16, **extra_kwargs
            ).to(DEVICE)
        except Exception as e:
            print(f'  FAILED to load {model_label}: {e}')
            results_table5[model_label] = {'direct': 100.0, 'json': -1, 'tool_call': -1, 'system': -1}
            continue

        # Independent direction extraction on extraction set, layer selection on validation
        directions = extract_direction(model, tokenizer, EXT_H, EXT_L)
        best_layer, best_auc = select_layer_by_auc(directions, model, tokenizer, VAL_H, VAL_L)
        direction = directions[best_layer]
        print(f'  Best layer: {best_layer} (AUC={best_auc:.4f} on validation)')

        # Compute direct gap on test set
        direct_gap, _, _, _ = compute_gap_and_auc(direction, best_layer, model, tokenizer, TST_H, TST_L)

        model_results = {'direct': 100.0}
        for fmt_name, fmt_fn in CROSS_MODEL_FORMATS.items():
            if fmt_name == 'direct':
                continue
            fmt_h = [fmt_fn(p) for p in TST_H]
            fmt_l = [fmt_fn(p) for p in TST_L]
            gap, auc, _, _ = compute_gap_and_auc(direction, best_layer, model, tokenizer, fmt_h, fmt_l)
            retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
            model_results[fmt_name] = retention
            print(f'  {fmt_name:<10} ret={retention:.1f}%')

        results_table5[model_label] = model_results

        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()

    return results_table5


def main():
    start = time.time()
    print("IEEE CARS 2026 - Full Experiment Pipeline (Stratified Splits)")
    print(f"Device: {DEVICE}")
    print(f"Splits: extraction={len(EXT_H)}, validation={len(VAL_H)}, test={len(TST_H)}")

    # Primary model experiments
    primary_results = run_primary_model()

    # Cross-model
    cross_results = run_cross_model(primary_results)
    primary_results['table5'] = cross_results

    # Save all results
    out_path = os.path.join(OUT_DIR, 'ieee_cars_stratified_results.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(primary_results, f, indent=2, default=convert)

    elapsed = time.time() - start
    print(f'\n{"="*70}')
    print(f'  COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} minutes)')
    print(f'  Results saved to: {out_path}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
