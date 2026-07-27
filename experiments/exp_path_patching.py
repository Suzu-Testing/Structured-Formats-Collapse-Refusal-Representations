"""Experiment 2: Path Patching (Component-Level).

Isolates whether refusal restoration comes from attention, MLPs, specific heads,
or specific token positions.

Four sub-experiments:
1. Attention-only patch: Replace only attention output at layer L
2. MLP-only patch: Replace only MLP output at layer L
3. Head-level patch: Patch individual attention heads
4. Token-position patch: Patch only format-token vs content-token positions

If patching attention restores refusal but MLP does not (or vice versa),
we identify WHICH computational component carries the safety signal.
If patching format-token positions restores refusal, format tokens are the
suppression site.
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    extract_refusal_direction, DEVICE
)
from exp_mechanistic_utils import (
    get_model_layers, fmt_direct, fmt_tool_call, fmt_system, fmt_json,
    classify_token_positions
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def get_hidden_at_layer(model, tokenizer, prompt, max_length=128):
    """Get hidden states at every layer for all token positions."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs


def collect_component_outputs(model, tokenizer, prompt, max_length=128):
    """Collect attention and MLP outputs at every layer."""
    model.eval()
    layers = get_model_layers(model)
    
    attn_outs = {}
    mlp_outs = {}
    hooks = []
    
    for i, layer in enumerate(layers):
        def make_attn_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    attn_outs[idx] = out[0].detach()
                else:
                    attn_outs[idx] = out.detach()
            return hook
        
        def make_mlp_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    mlp_outs[idx] = out[0].detach()
                else:
                    mlp_outs[idx] = out.detach()
            return hook
        
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    for h in hooks:
        h.remove()
    
    hidden_states = outputs.hidden_states
    return attn_outs, mlp_outs, hidden_states, inputs


def patch_attn_only(model, tokenizer, source_prompt, target_prompt,
                    refusal_direction, layer_idx, max_length=128):
    """Patch ONLY the attention output from source into target at layer_idx."""
    layers = get_model_layers(model)
    
    # Get source attention output
    src_attn_out = {}
    hooks = []
    
    def src_hook(module, inp, out):
        if isinstance(out, tuple):
            src_attn_out[0] = out[0].detach()
        else:
            src_attn_out[0] = out.detach()
    
    hooks.append(layers[layer_idx].self_attn.register_forward_hook(src_hook))
    src_inputs = tokenizer(source_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        model(**src_inputs)
    for h in hooks:
        h.remove()
    hooks = []
    
    # Patch into target
    def patch_hook(module, inp, out):
        src = src_attn_out[0]
        if isinstance(out, tuple):
            tgt = out[0].clone()
            min_seq = min(tgt.shape[1], src.shape[1])
            min_dim = min(tgt.shape[2], src.shape[2])
            tgt[0, -1, :min_dim] = src[0, -1, :min_dim]  # Patch last token only
            return (tgt,) + out[1:]
        else:
            tgt = out.clone()
            min_dim = min(tgt.shape[2], src.shape[2])
            tgt[0, -1, :min_dim] = src[0, -1, :min_dim]
            return tgt
    
    hooks.append(layers[layer_idx].self_attn.register_forward_hook(patch_hook))
    tgt_inputs = tokenizer(target_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tgt_inputs, output_hidden_states=True)
    for h in hooks:
        h.remove()
    
    final = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
    return float(np.dot(final, refusal_direction / (np.linalg.norm(refusal_direction) + 1e-10)))


def patch_mlp_only(model, tokenizer, source_prompt, target_prompt,
                   refusal_direction, layer_idx, max_length=128):
    """Patch ONLY the MLP output from source into target at layer_idx."""
    layers = get_model_layers(model)
    
    src_mlp_out = {}
    hooks = []
    
    def src_hook(module, inp, out):
        if isinstance(out, tuple):
            src_mlp_out[0] = out[0].detach()
        else:
            src_mlp_out[0] = out.detach()
    
    hooks.append(layers[layer_idx].mlp.register_forward_hook(src_hook))
    src_inputs = tokenizer(source_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        model(**src_inputs)
    for h in hooks:
        h.remove()
    hooks = []
    
    def patch_hook(module, inp, out):
        src = src_mlp_out[0]
        if isinstance(out, tuple):
            tgt = out[0].clone()
            min_dim = min(tgt.shape[2], src.shape[2])
            tgt[0, -1, :min_dim] = src[0, -1, :min_dim]
            return (tgt,) + out[1:]
        else:
            tgt = out.clone()
            min_dim = min(tgt.shape[2], src.shape[2])
            tgt[0, -1, :min_dim] = src[0, -1, :min_dim]
            return tgt
    
    hooks.append(layers[layer_idx].mlp.register_forward_hook(patch_hook))
    tgt_inputs = tokenizer(target_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tgt_inputs, output_hidden_states=True)
    for h in hooks:
        h.remove()
    
    final = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
    return float(np.dot(final, refusal_direction / (np.linalg.norm(refusal_direction) + 1e-10)))


def patch_position_only(model, tokenizer, source_prompt, target_prompt,
                        refusal_direction, layer_idx, positions, max_length=128):
    """Patch full hidden state but ONLY at specific token positions."""
    layers = get_model_layers(model)
    
    # Get source full hidden at this layer
    src_inputs = tokenizer(source_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        src_outputs = model(**src_inputs, output_hidden_states=True)
    src_hidden = src_outputs.hidden_states[layer_idx + 1].detach()
    
    hooks = []
    
    def patch_hook(module, inp, out):
        if isinstance(out, tuple):
            tgt = out[0].clone()
        else:
            tgt = out.clone()
        
        for pos in positions:
            if pos < tgt.shape[1] and pos < src_hidden.shape[1]:
                min_dim = min(tgt.shape[2], src_hidden.shape[2])
                tgt[0, pos, :min_dim] = src_hidden[0, pos, :min_dim]
        
        if isinstance(out, tuple):
            return (tgt,) + out[1:]
        return tgt
    
    hooks.append(layers[layer_idx].register_forward_hook(patch_hook))
    tgt_inputs = tokenizer(target_prompt, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tgt_inputs, output_hidden_states=True)
    for h in hooks:
        h.remove()
    
    final = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
    return float(np.dot(final, refusal_direction / (np.linalg.norm(refusal_direction) + 1e-10)))


def main():
    print('='*70, flush=True)
    print('EXPERIMENT 2: PATH PATCHING (COMPONENT-LEVEL)', flush=True)
    print('Which component carries the safety signal?', flush=True)
    print('='*70, flush=True)
    
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
    print(f'  Loaded.', flush=True)
    
    # Extract refusal direction
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 27
    direction = refusal_dirs[best_layer]
    layers = get_model_layers(model)
    n_layers = len(layers)
    print(f'  Best layer: {best_layer}, total layers: {n_layers}', flush=True)
    
    N = 8
    test_layers = list(range(0, n_layers, 3)) + [best_layer]  # Every 3rd layer + best
    test_layers = sorted(set(test_layers))
    
    results = []
    
    # ============================================================
    # BASELINES
    # ============================================================
    print(f'\n--- Computing baselines ---', flush=True)
    
    direct_projs = []
    for p in HARMFUL_PROMPTS[:N]:
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].cpu().numpy()
        direct_projs.append(float(np.dot(h, direction / (np.linalg.norm(direction) + 1e-10))))
    direct_mean = np.mean(direct_projs)
    
    formatted_projs = {}
    for fmt_name, fmt_fn in [('tool_call', fmt_tool_call), ('system', fmt_system)]:
        projs = []
        for p in HARMFUL_PROMPTS[:N]:
            inputs = tokenizer(fmt_fn(p), return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].cpu().numpy()
            projs.append(float(np.dot(h, direction / (np.linalg.norm(direction) + 1e-10))))
        formatted_projs[fmt_name] = np.mean(projs)
    
    print(f'  Direct mean projection: {direct_mean:.4f}', flush=True)
    for fmt, proj in formatted_projs.items():
        print(f'  {fmt} mean projection: {proj:.4f}', flush=True)
    
    # ============================================================
    # SUB-EXP 1: Attention-only vs MLP-only patching
    # ============================================================
    print(f'\n--- SUB-EXP 1: Attention vs MLP patching ---', flush=True)
    
    for fmt_name, fmt_fn in [('tool_call', fmt_tool_call), ('system', fmt_system)]:
        baseline = formatted_projs[fmt_name]
        gap = direct_mean - baseline
        
        print(f'\n  Format: {fmt_name} (gap to restore: {gap:.4f})', flush=True)
        
        for layer_idx in test_layers:
            # Attention-only patch
            attn_projs = []
            mlp_projs = []
            
            for p in HARMFUL_PROMPTS[:N]:
                try:
                    attn_proj = patch_attn_only(model, tokenizer, p, fmt_fn(p),
                                               direction, layer_idx)
                    attn_projs.append(attn_proj)
                except:
                    attn_projs.append(baseline)
                
                try:
                    mlp_proj = patch_mlp_only(model, tokenizer, p, fmt_fn(p),
                                            direction, layer_idx)
                    mlp_projs.append(mlp_proj)
                except:
                    mlp_projs.append(baseline)
            
            attn_mean = np.mean(attn_projs)
            mlp_mean = np.mean(mlp_projs)
            
            attn_restoration = ((attn_mean - baseline) / gap * 100) if gap != 0 else 0
            mlp_restoration = ((mlp_mean - baseline) / gap * 100) if gap != 0 else 0
            
            results.append({
                'format': fmt_name,
                'layer': layer_idx,
                'component': 'attention',
                'mean_projection': attn_mean,
                'restoration_pct': attn_restoration,
            })
            results.append({
                'format': fmt_name,
                'layer': layer_idx,
                'component': 'mlp',
                'mean_projection': mlp_mean,
                'restoration_pct': mlp_restoration,
            })
            
            print(f'    Layer {layer_idx:>2d}: attn={attn_restoration:>6.1f}%, mlp={mlp_restoration:>6.1f}%', flush=True)
    
    # ============================================================
    # SUB-EXP 2: Position patching (format vs content tokens)
    # ============================================================
    print(f'\n--- SUB-EXP 2: Position patching (format vs content tokens) ---', flush=True)
    
    critical_layers = [best_layer, best_layer - 3, best_layer - 6]
    critical_layers = [l for l in critical_layers if 0 <= l < n_layers]
    
    for fmt_name, fmt_fn in [('tool_call', fmt_tool_call)]:
        baseline = formatted_projs[fmt_name]
        gap = direct_mean - baseline
        
        print(f'\n  Format: {fmt_name}', flush=True)
        
        for layer_idx in critical_layers:
            format_pos_projs = []
            content_pos_projs = []
            last_pos_projs = []
            
            for p in HARMFUL_PROMPTS[:N]:
                formatted = fmt_fn(p)
                tokens = tokenizer.tokenize(formatted)
                labels = classify_token_positions(tokens, p, fmt_name)
                
                format_positions = [i for i, l in enumerate(labels) if l == 'format']
                content_positions = [i for i, l in enumerate(labels) if l == 'content']
                last_position = [len(tokens) - 1] if tokens else [0]
                
                try:
                    fp = patch_position_only(model, tokenizer, p, formatted,
                                           direction, layer_idx, format_positions)
                    format_pos_projs.append(fp)
                except:
                    format_pos_projs.append(baseline)
                
                try:
                    cp = patch_position_only(model, tokenizer, p, formatted,
                                           direction, layer_idx, content_positions)
                    content_pos_projs.append(cp)
                except:
                    content_pos_projs.append(baseline)
                
                try:
                    lp = patch_position_only(model, tokenizer, p, formatted,
                                           direction, layer_idx, last_position)
                    last_pos_projs.append(lp)
                except:
                    last_pos_projs.append(baseline)
            
            fp_mean = np.mean(format_pos_projs)
            cp_mean = np.mean(content_pos_projs)
            lp_mean = np.mean(last_pos_projs)
            
            fp_rest = ((fp_mean - baseline) / gap * 100) if gap != 0 else 0
            cp_rest = ((cp_mean - baseline) / gap * 100) if gap != 0 else 0
            lp_rest = ((lp_mean - baseline) / gap * 100) if gap != 0 else 0
            
            results.append({'format': fmt_name, 'layer': layer_idx,
                           'component': 'position_format', 'mean_projection': fp_mean,
                           'restoration_pct': fp_rest})
            results.append({'format': fmt_name, 'layer': layer_idx,
                           'component': 'position_content', 'mean_projection': cp_mean,
                           'restoration_pct': cp_rest})
            results.append({'format': fmt_name, 'layer': layer_idx,
                           'component': 'position_last', 'mean_projection': lp_mean,
                           'restoration_pct': lp_rest})
            
            print(f'    Layer {layer_idx:>2d}: format_pos={fp_rest:>6.1f}%, '
                  f'content_pos={cp_rest:>6.1f}%, last_pos={lp_rest:>6.1f}%', flush=True)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('PATH PATCHING SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    df = pd.DataFrame(results)
    
    # Best restoration per component
    for component in ['attention', 'mlp', 'position_format', 'position_content', 'position_last']:
        comp_df = df[df['component'] == component]
        if len(comp_df) > 0:
            best = comp_df.loc[comp_df['restoration_pct'].idxmax()]
            print(f'  {component:<20}: best restoration = {best["restoration_pct"]:.1f}% at layer {int(best["layer"])}', flush=True)
    
    # Key finding
    attn_best = df[df['component'] == 'attention']['restoration_pct'].max()
    mlp_best = df[df['component'] == 'mlp']['restoration_pct'].max()
    
    print(f'\n  INTERPRETATION:', flush=True)
    if attn_best > mlp_best + 20:
        print(f'  ATTENTION dominates: safety signal flows through attention heads.', flush=True)
        print(f'  Format tokens likely suppress safety by redirecting attention.', flush=True)
    elif mlp_best > attn_best + 20:
        print(f'  MLP dominates: safety signal is in feedforward computation.', flush=True)
        print(f'  Format tokens likely suppress safety by altering MLP inputs.', flush=True)
    else:
        print(f'  Both components contribute. Safety signal uses full residual stream.', flush=True)
    
    fp_best = df[df['component'] == 'position_format']['restoration_pct'].max() if len(df[df['component'] == 'position_format']) > 0 else 0
    cp_best = df[df['component'] == 'position_content']['restoration_pct'].max() if len(df[df['component'] == 'position_content']) > 0 else 0
    
    if fp_best > cp_best + 15:
        print(f'  FORMAT POSITIONS are the suppression site ({fp_best:.1f}% vs content {cp_best:.1f}%).', flush=True)
        print(f'  Patching format-token states restores refusal -> format tokens actively suppress.', flush=True)
    elif cp_best > fp_best + 15:
        print(f'  CONTENT POSITIONS carry the suppressed signal ({cp_best:.1f}% vs format {fp_best:.1f}%).', flush=True)
    
    # Save
    outpath = os.path.join(OUT_DIR, 'exp_path_patching.csv')
    df.to_csv(outpath, index=False)
    print(f'\nResults saved to: {outpath}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
