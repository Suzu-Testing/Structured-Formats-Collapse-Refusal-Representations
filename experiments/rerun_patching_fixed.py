"""Rerun patching experiments with corrected hidden_states indexing.

FIX: All projections now use hidden_states[best_layer] consistently.
The bug was: patched projections used hidden_states[best_layer + 1] while
direction extraction and baselines used hidden_states[best_layer].

In HuggingFace, hidden_states[0] = embedding, hidden_states[i] = output of
transformer block (i-1). So hidden_states[26] = output of block 25.
Using best_layer=26 everywhere ensures apples-to-apples comparison.

Outputs:
  - Table IV: Cumulative activation patching restoration (%)
  - Component-level: Attention vs MLP at Layer 0
"""

import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)

(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)

FORMATS = {
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}


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


def extract_direction(model, tokenizer, harmful_prompts, harmless_prompts):
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
    h_states = get_hidden_states(model, tokenizer, harmful_prompts)
    l_states = get_hidden_states(model, tokenizer, harmless_prompts)
    best_auc = 0
    best_layer = 0
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


def main():
    print("=" * 70)
    print("PATCHING RERUN - CORRECTED hidden_states INDEXING")
    print("=" * 70)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    print('\nExtracting direction (extraction set, N=20)...')
    directions = extract_direction(model, tokenizer, EXT_H, EXT_L)

    print('Selecting layer by AUC (validation set, N=20)...')
    best_layer, best_auc = select_layer_by_auc(directions, model, tokenizer, VAL_H, VAL_L)
    print(f'  best_layer = {best_layer}, AUC = {best_auc:.4f}')
    direction = directions[best_layer]

    # ================================================================
    # TABLE IV: Cumulative activation patching (N=8)
    # ================================================================
    print(f'\n{"="*70}')
    print("TABLE IV: Cumulative Activation Patching (N=8)")
    print(f'  All projections use hidden_states[{best_layer}]')
    print(f'{"="*70}')

    patch_prompts = TST_H[:8]
    patch_layers = [0, 6, 12, 18, 24, 26]
    results_table4 = {}

    for fmt_name, fmt_fn in FORMATS.items():
        print(f'\n  --- {fmt_name} ---')
        restorations = []
        for L in patch_layers:
            layer_restorations = []
            for p in patch_prompts:
                direct_prompt = p
                formatted_prompt = fmt_fn(p)

                d_states = get_hidden_states(model, tokenizer, [direct_prompt])
                direct_proj = np.dot(d_states[0][best_layer], direction)

                f_states = get_hidden_states(model, tokenizer, [formatted_prompt])
                formatted_proj = np.dot(f_states[0][best_layer], direction)

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

                # FIXED: use hidden_states[best_layer], NOT best_layer + 1
                patched_proj = float(np.dot(
                    patched_out.hidden_states[best_layer][0, -1, :].cpu().numpy(),
                    direction
                ))

                denom = direct_proj - formatted_proj
                if abs(denom) > 1e-8:
                    restoration = (patched_proj - formatted_proj) / denom * 100
                else:
                    restoration = 0
                layer_restorations.append(restoration)

            mean_rest = np.mean(layer_restorations)
            restorations.append(mean_rest)
            print(f'    L0..{L:>2d}: {mean_rest:>7.1f}%')

        results_table4[fmt_name] = dict(zip(patch_layers, restorations))

    # ================================================================
    # COMPONENT-LEVEL: Attention vs MLP at Layer 0
    # ================================================================
    print(f'\n{"="*70}')
    print("COMPONENT-LEVEL: Attention vs MLP patching at Layer 0 (N=8)")
    print(f'  All projections use hidden_states[{best_layer}]')
    print(f'{"="*70}')

    PATCH_LAYER = 0
    component_results = {}

    for fmt_name, fmt_fn in FORMATS.items():
        attn_restorations = []
        mlp_restorations = []

        for p in patch_prompts:
            direct_prompt = p
            formatted_prompt = fmt_fn(p)

            d_inputs = tokenizer(direct_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            f_inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

            with torch.no_grad():
                d_out = model(**d_inputs, output_hidden_states=True)
                direct_proj = float(np.dot(d_out.hidden_states[best_layer][0, -1, :].cpu().numpy(), direction))

                f_out = model(**f_inputs, output_hidden_states=True)
                formatted_proj = float(np.dot(f_out.hidden_states[best_layer][0, -1, :].cpu().numpy(), direction))

            denom = direct_proj - formatted_proj
            if abs(denom) < 1e-8:
                continue

            # Attention-only patching
            attn_output_direct = None
            def capture_attn_hook(module, input, output):
                nonlocal attn_output_direct
                attn_output_direct = output[0][0, -1, :].clone()

            hook = model.model.layers[PATCH_LAYER].self_attn.register_forward_hook(capture_attn_hook)
            with torch.no_grad():
                model(**d_inputs, output_hidden_states=True)
            hook.remove()

            def patch_attn_hook(module, input, output):
                patched = list(output)
                h = patched[0].clone()
                h[0, -1, :] = attn_output_direct
                patched[0] = h
                return tuple(patched)

            hook = model.model.layers[PATCH_LAYER].self_attn.register_forward_hook(patch_attn_hook)
            with torch.no_grad():
                patched_out = model(**f_inputs, output_hidden_states=True)
            hook.remove()

            patched_proj = float(np.dot(patched_out.hidden_states[best_layer][0, -1, :].cpu().numpy(), direction))
            attn_restorations.append((patched_proj - formatted_proj) / denom * 100)

            # MLP-only patching
            mlp_output_direct = None
            def capture_mlp_hook(module, input, output):
                nonlocal mlp_output_direct
                mlp_output_direct = output[0, -1, :].clone() if output.dim() == 3 else output.clone()

            hook = model.model.layers[PATCH_LAYER].mlp.register_forward_hook(capture_mlp_hook)
            with torch.no_grad():
                model(**d_inputs, output_hidden_states=True)
            hook.remove()

            def patch_mlp_hook(module, input, output):
                patched = output.clone()
                if patched.dim() == 3:
                    patched[0, -1, :] = mlp_output_direct
                return patched

            hook = model.model.layers[PATCH_LAYER].mlp.register_forward_hook(patch_mlp_hook)
            with torch.no_grad():
                patched_out = model(**f_inputs, output_hidden_states=True)
            hook.remove()

            patched_proj = float(np.dot(patched_out.hidden_states[best_layer][0, -1, :].cpu().numpy(), direction))
            mlp_restorations.append((patched_proj - formatted_proj) / denom * 100)

        attn_mean = np.mean(attn_restorations) if attn_restorations else 0
        mlp_mean = np.mean(mlp_restorations) if mlp_restorations else 0
        print(f'\n  {fmt_name}:')
        print(f'    Attention-only L0: {attn_mean:.1f}% (std={np.std(attn_restorations):.1f}, N={len(attn_restorations)})')
        print(f'    MLP-only L0:       {mlp_mean:.1f}% (std={np.std(mlp_restorations):.1f}, N={len(mlp_restorations)})')
        component_results[fmt_name] = {
            'attn_mean': attn_mean, 'attn_std': float(np.std(attn_restorations)),
            'mlp_mean': mlp_mean, 'mlp_std': float(np.std(mlp_restorations)),
            'n': len(attn_restorations)
        }

    # ================================================================
    # ATTENTION ROUTING DIVERGENCE (observational, no patching)
    # ================================================================
    print(f'\n{"="*70}')
    print("ATTENTION ROUTING DIVERGENCE at Layer 0 (N=15)")
    print(f'{"="*70}')

    routing_prompts = TST_H[:15]
    divergences = []

    for p in routing_prompts:
        direct_prompt = p
        formatted_prompt = FORMATS['tool_call'](p)

        attn_weights_direct = None
        attn_weights_formatted = None

        def capture_attn_weights_direct(module, input, output):
            nonlocal attn_weights_direct
            if hasattr(module, '_attn_weights'):
                attn_weights_direct = module._attn_weights

        def capture_attn_weights_formatted(module, input, output):
            nonlocal attn_weights_formatted
            if hasattr(module, '_attn_weights'):
                attn_weights_formatted = module._attn_weights

        d_inputs = tokenizer(direct_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        f_inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

        with torch.no_grad():
            d_out = model(**d_inputs, output_hidden_states=True, output_attentions=True)
            f_out = model(**f_inputs, output_hidden_states=True, output_attentions=True)

        if d_out.attentions is not None and f_out.attentions is not None:
            d_attn = d_out.attentions[0][0, :, -1, :]  # Layer 0, all heads, last query pos
            f_attn = f_out.attentions[0][0, :, -1, :]

            d_len = d_attn.shape[-1]
            f_len = f_attn.shape[-1]

            for head_idx in range(d_attn.shape[0]):
                d_head = d_attn[head_idx].cpu().numpy()
                f_head = f_attn[head_idx, :f_len].cpu().numpy()

                d_content_mass = d_head.sum()
                f_format_tokens = max(0, f_len - d_len)
                if f_format_tokens > 0 and f_len > 0:
                    f_format_mass = f_head[:f_format_tokens].sum()
                    f_content_mass = f_head[f_format_tokens:].sum()
                    d_dist = np.array([0.0, d_content_mass])
                    f_dist = np.array([f_format_mass, f_content_mass])
                    l1 = np.sum(np.abs(d_dist - f_dist))
                    if l1 > 0.5:
                        divergences.append(l1)

    if divergences:
        print(f'  Valid heads with divergence > 0.5: {len(divergences)}')
        print(f'  Mean L1 divergence: {np.mean(divergences):.2f}')
        print(f'  Range: {np.min(divergences):.2f} to {np.max(divergences):.2f}')
    else:
        print('  No valid divergence measurements (attention weights may not be accessible)')

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results = {
        'best_layer': best_layer,
        'best_layer_auc': float(best_auc),
        'indexing_note': f'All projections use hidden_states[{best_layer}] consistently',
        'table4_cumulative_patching': results_table4,
        'component_level': component_results,
        'routing_divergence': {
            'n_measurements': len(divergences),
            'mean': float(np.mean(divergences)) if divergences else None,
            'min': float(np.min(divergences)) if divergences else None,
            'max': float(np.max(divergences)) if divergences else None,
        }
    }

    out_path = os.path.join(OUT_DIR, 'patching_fixed_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out_path}')
    print("DONE.")


if __name__ == '__main__':
    main()
