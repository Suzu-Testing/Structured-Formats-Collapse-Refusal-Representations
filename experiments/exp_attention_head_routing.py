"""Experiment 1: Attention-Head Routing Test.

Measures whether harmful content in tool_call/system format attends through
different heads than direct harmful content.

Looks for heads where:
  - direct harmful -> safety-relevant attention pattern (attends to content tokens)
  - tool_call harmful -> format/control-token attention pattern (attends to format tokens)

If specific heads route disproportionately through system/tool tokens, that
supports "privileged channel" not just "distribution shift."

Metrics:
- Per-head "routing divergence": KL(attention_direct || attention_toolcall)
- Content-vs-format attention ratio per head
- Identification of "safety heads" vs "format-routing heads"
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from scipy.special import kl_div, rel_entr
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, DEVICE
)
from exp_mechanistic_utils import (
    get_attention_by_head, classify_token_positions,
    fmt_direct, fmt_tool_call, fmt_system, fmt_json
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def compute_attention_to_content_vs_format(attn_map, token_labels):
    """Compute fraction of last-token attention going to content vs format tokens.
    
    Args:
        attn_map: (n_heads, seq_len, seq_len) attention tensor
        token_labels: list of 'content' or 'format' labels per position
    
    Returns: per-head (content_frac, format_frac) arrays
    """
    n_heads = attn_map.shape[0]
    seq_len = attn_map.shape[1]
    
    content_mask = np.array([1.0 if l == 'content' else 0.0 for l in token_labels[:seq_len]])
    format_mask = np.array([1.0 if l == 'format' else 0.0 for l in token_labels[:seq_len]])
    
    last_token_attn = attn_map[:, -1, :seq_len].numpy()  # (n_heads, seq_len)
    
    content_attn = (last_token_attn * content_mask).sum(axis=1)  # (n_heads,)
    format_attn = (last_token_attn * format_mask).sum(axis=1)    # (n_heads,)
    
    return content_attn, format_attn


def compute_routing_divergence(attn_direct, attn_formatted, eps=1e-8):
    """Compute KL divergence between attention distributions per head.
    
    Both inputs: (n_heads, seq_len) last-token attention distributions.
    Since sequences may differ in length, compare entropy and content-ratio instead.
    """
    n_heads = attn_direct.shape[0]
    
    entropy_direct = -np.sum(attn_direct * np.log(attn_direct + eps), axis=1)
    entropy_formatted = -np.sum(attn_formatted * np.log(attn_formatted + eps), axis=1)
    
    entropy_diff = entropy_formatted - entropy_direct
    
    return entropy_diff


def main():
    print('='*70, flush=True)
    print('EXPERIMENT 1: ATTENTION-HEAD ROUTING TEST', flush=True)
    print('Do tool_call formats route through different heads than direct?', flush=True)
    print('='*70, flush=True)
    
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', attn_implementation='eager')
    print(f'  Loaded (eager attention for output_attentions=True).', flush=True)
    
    N = 15  # Prompts to analyze
    formats = {
        'direct': fmt_direct,
        'tool_call': fmt_tool_call,
        'system': fmt_system,
        'json': fmt_json,
    }
    
    # ============================================================
    # STEP 1: Collect per-head attention patterns
    # ============================================================
    print(f'\n--- STEP 1: Collecting attention patterns (N={N}) ---', flush=True)
    
    # Storage: per format, per layer, per head -> content_frac, format_frac
    head_stats = {}  # (format, layer, head) -> list of content_fracs
    
    for fmt_name, fmt_fn in formats.items():
        print(f'  Processing format: {fmt_name}', flush=True)
        head_stats[fmt_name] = {}
        
        for prompt_idx, prompt in enumerate(HARMFUL_PROMPTS[:N]):
            formatted = fmt_fn(prompt)
            
            attn_maps, tokens = get_attention_by_head(model, tokenizer, formatted)
            token_labels = classify_token_positions(tokens, prompt, fmt_name)
            
            for layer_idx, attn in attn_maps.items():
                if layer_idx not in head_stats[fmt_name]:
                    head_stats[fmt_name][layer_idx] = {'content': [], 'format': []}
                
                content_attn, format_attn = compute_attention_to_content_vs_format(
                    attn, token_labels)
                
                head_stats[fmt_name][layer_idx]['content'].append(content_attn)
                head_stats[fmt_name][layer_idx]['format'].append(format_attn)
    
    # ============================================================
    # STEP 2: Compute per-head routing divergence
    # ============================================================
    print(f'\n--- STEP 2: Computing routing divergence ---', flush=True)
    
    results = []
    n_layers = len(head_stats['direct'])
    
    for layer_idx in range(n_layers):
        if layer_idx not in head_stats['direct']:
            continue
        
        # Mean content attention for direct format
        direct_content = np.mean(head_stats['direct'][layer_idx]['content'], axis=0)  # (n_heads,)
        direct_format = np.mean(head_stats['direct'][layer_idx]['format'], axis=0)
        
        n_heads = len(direct_content)
        
        for fmt_name in ['tool_call', 'system', 'json']:
            if layer_idx not in head_stats[fmt_name]:
                continue
            
            fmt_content = np.mean(head_stats[fmt_name][layer_idx]['content'], axis=0)
            fmt_format = np.mean(head_stats[fmt_name][layer_idx]['format'], axis=0)
            
            for head_idx in range(n_heads):
                # Routing shift: how much does this head shift from content to format?
                content_shift = direct_content[head_idx] - fmt_content[head_idx]
                format_shift = fmt_format[head_idx] - direct_format[head_idx]
                
                # A head is "routing-divergent" if it attends to content in direct
                # but to format tokens in tool_call
                routing_score = content_shift + format_shift
                
                results.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'format': fmt_name,
                    'direct_content_attn': float(direct_content[head_idx]),
                    'direct_format_attn': float(direct_format[head_idx]),
                    'formatted_content_attn': float(fmt_content[head_idx]),
                    'formatted_format_attn': float(fmt_format[head_idx]),
                    'content_shift': float(content_shift),
                    'format_shift': float(format_shift),
                    'routing_divergence': float(routing_score),
                })
    
    df = pd.DataFrame(results)
    
    # ============================================================
    # STEP 3: Identify safety-relevant and format-routing heads
    # ============================================================
    print(f'\n--- STEP 3: Head classification ---', flush=True)
    
    # Top routing-divergent heads for tool_call
    tc_df = df[df['format'] == 'tool_call'].copy()
    tc_df_sorted = tc_df.sort_values('routing_divergence', ascending=False)
    
    print(f'\n  TOP 10 ROUTING-DIVERGENT HEADS (tool_call):', flush=True)
    print(f'  {"Layer":<6} {"Head":<6} {"Routing Div":<12} {"Direct->Content":<16} {"TC->Format":<12}', flush=True)
    print(f'  {"-"*52}', flush=True)
    
    for _, row in tc_df_sorted.head(10).iterrows():
        print(f'  {int(row["layer"]):<6} {int(row["head"]):<6} {row["routing_divergence"]:>8.4f}    '
              f'{row["direct_content_attn"]:>8.4f}        {row["formatted_format_attn"]:>8.4f}', flush=True)
    
    # Safety heads: high content attention in direct format
    safety_heads = tc_df[tc_df['direct_content_attn'] > tc_df['direct_content_attn'].quantile(0.9)]
    print(f'\n  SAFETY HEADS (top 10% content attention in direct):', flush=True)
    print(f'  {len(safety_heads)} heads identified', flush=True)
    for _, row in safety_heads.sort_values('routing_divergence', ascending=False).head(5).iterrows():
        print(f'    Layer {int(row["layer"])}, Head {int(row["head"])}: '
              f'content_attn={row["direct_content_attn"]:.4f}, routing_div={row["routing_divergence"]:.4f}', flush=True)
    
    # Format-routing heads: high format attention in tool_call but low in direct
    format_heads = tc_df[(tc_df['formatted_format_attn'] > tc_df['formatted_format_attn'].quantile(0.9)) &
                         (tc_df['direct_format_attn'] < tc_df['direct_format_attn'].quantile(0.5))]
    print(f'\n  FORMAT-ROUTING HEADS (high format attn in TC, low in direct):', flush=True)
    print(f'  {len(format_heads)} heads identified', flush=True)
    for _, row in format_heads.sort_values('formatted_format_attn', ascending=False).head(5).iterrows():
        print(f'    Layer {int(row["layer"])}, Head {int(row["head"])}: '
              f'tc_format_attn={row["formatted_format_attn"]:.4f}, direct_format={row["direct_format_attn"]:.4f}', flush=True)
    
    # ============================================================
    # STEP 4: Layer-wise routing divergence summary
    # ============================================================
    print(f'\n--- STEP 4: Layer-wise summary ---', flush=True)
    
    layer_summary = tc_df.groupby('layer').agg({
        'routing_divergence': ['mean', 'max'],
        'content_shift': 'mean',
        'format_shift': 'mean',
    }).reset_index()
    
    print(f'\n  {"Layer":<6} {"Mean Route Div":<15} {"Max Route Div":<15} {"Content Shift":<15} {"Format Shift":<15}', flush=True)
    print(f'  {"-"*66}', flush=True)
    
    for _, row in layer_summary.iterrows():
        layer = int(row[('layer', '')])
        mean_rd = row[('routing_divergence', 'mean')]
        max_rd = row[('routing_divergence', 'max')]
        cs = row[('content_shift', 'mean')]
        fs = row[('format_shift', 'mean')]
        marker = ' ***' if mean_rd > layer_summary[('routing_divergence', 'mean')].quantile(0.8) else ''
        print(f'  {layer:<6} {mean_rd:<15.4f} {max_rd:<15.4f} {cs:<15.4f} {fs:<15.4f}{marker}', flush=True)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('ATTENTION-HEAD ROUTING SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    mean_routing_div = tc_df['routing_divergence'].mean()
    max_routing_div = tc_df['routing_divergence'].max()
    pct_shifted = (tc_df['routing_divergence'] > 0.05).mean() * 100
    
    print(f'\n  Mean routing divergence (tool_call): {mean_routing_div:.4f}', flush=True)
    print(f'  Max routing divergence: {max_routing_div:.4f}', flush=True)
    print(f'  Percentage of heads with routing_div > 0.05: {pct_shifted:.1f}%', flush=True)
    
    # Cross-format consistency
    for fmt in ['tool_call', 'system', 'json']:
        fmt_df = df[df['format'] == fmt]
        mean_rd = fmt_df['routing_divergence'].mean()
        print(f'  {fmt}: mean routing divergence = {mean_rd:.4f}', flush=True)
    
    print(f'\n  INTERPRETATION:', flush=True)
    if mean_routing_div > 0.02:
        print(f'  Significant routing divergence detected.', flush=True)
        print(f'  Format tokens create a distinct computational route through specific heads.', flush=True)
        print(f'  This supports "privileged channel" over "mere distribution shift."', flush=True)
    else:
        print(f'  Routing divergence is weak. Format suppression may operate through', flush=True)
        print(f'  representation content rather than attention routing.', flush=True)
    
    # Save
    outpath = os.path.join(OUT_DIR, 'exp_attention_head_routing.csv')
    df.to_csv(outpath, index=False)
    print(f'\nResults saved to: {outpath}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
