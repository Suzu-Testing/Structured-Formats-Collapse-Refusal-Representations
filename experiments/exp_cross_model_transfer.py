"""Cross-Model Transfer: Does the refusal geometry generalize?

Train probe/extract direction on Qwen2.5-1.5B, test on:
- TinyLlama-1.1B
- Phi-2
- Qwen2.5-3B

Questions:
1. Does the format-dependent encoding transfer across architectures?
2. Does recalibration transfer?
3. Is this a universal phenomenon or model-specific?

Either result is publishable:
- YES transfer: phenomenon is fundamental to instruction tuning
- NO transfer: format-local safety geometry (still interesting)
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def get_hidden_states(model, tokenizer, prompts, max_length=128):
    """Extract hidden states at last token for all prompts."""
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                             max_length=max_length, padding=False).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_token_hidden = [h[0, -1, :].cpu().numpy() for h in hidden_states]
            all_hidden.append(last_token_hidden)
    return all_hidden


def extract_refusal_direction(model, tokenizer, harmful_prompts, harmless_prompts):
    """Extract refusal direction via difference-in-means."""
    print('    Extracting hidden states...', flush=True)
    harmful_hidden = get_hidden_states(model, tokenizer, harmful_prompts)
    harmless_hidden = get_hidden_states(model, tokenizer, harmless_prompts)
    
    n_layers = len(harmful_hidden[0])
    directions = []
    
    for layer in range(n_layers):
        harmful_mean = np.mean([h[layer] for h in harmful_hidden], axis=0)
        harmless_mean = np.mean([h[layer] for h in harmless_hidden], axis=0)
        direction = harmful_mean - harmless_mean
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions.append(direction)
    
    return directions


def compute_retention(model, tokenizer, direction, best_layer, fmt_fn, N=20):
    """Compute retention for a given format."""
    harmful_formatted = [fmt_fn(p) for p in HARMFUL_PROMPTS[:N]]
    harmless_formatted = [fmt_fn(p) for p in HARMLESS_PROMPTS[:N]]
    
    harmful_h = get_hidden_states(model, tokenizer, harmful_formatted)
    harmless_h = get_hidden_states(model, tokenizer, harmless_formatted)
    
    harmful_projs = [float(np.dot(h[best_layer], direction)) for h in harmful_h]
    harmless_projs = [float(np.dot(h[best_layer], direction)) for h in harmless_h]
    
    gap = np.mean(harmful_projs) - np.mean(harmless_projs)
    auc = roc_auc_score(
        [1]*len(harmful_projs) + [0]*len(harmless_projs),
        harmful_projs + harmless_projs
    )
    
    return gap, auc, harmful_projs, harmless_projs


FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}


def main():
    print('='*70, flush=True)
    print('CROSS-MODEL TRANSFER: DOES FORMAT-DEPENDENT ENCODING GENERALIZE?', flush=True)
    print('='*70, flush=True)

    models = [
        ('Qwen/Qwen2.5-1.5B-Instruct', 27),
        ('Qwen/Qwen2.5-3B-Instruct', 35),
        ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 21),
        ('microsoft/phi-2', 31),
    ]
    
    results = []
    model_directions = {}
    model_probes = {}

    # Phase 1: Extract directions and train probes for each model
    for model_name, best_layer in models:
        short_name = model_name.split('/')[-1]
        print(f'\n  Loading: {short_name}', flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16,
            device_map='auto')
        
        # Extract direction
        directions = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
        direction = directions[best_layer]
        model_directions[short_name] = (direction, best_layer)
        
        # Train probe on direct format
        harmful_h = get_hidden_states(model, tokenizer, HARMFUL_PROMPTS)
        harmless_h = get_hidden_states(model, tokenizer, HARMLESS_PROMPTS)
        
        X = np.array([h[best_layer] for h in harmful_h + harmless_h])
        y = np.array([1]*len(HARMFUL_PROMPTS) + [0]*len(HARMLESS_PROMPTS))
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        model_probes[short_name] = lr
        
        # Measure within-model format retention
        direct_gap, _, _, _ = compute_retention(model, tokenizer, direction, best_layer, FORMATS['direct'])
        
        for fmt_name, fmt_fn in FORMATS.items():
            gap, auc, h_projs, hl_projs = compute_retention(
                model, tokenizer, direction, best_layer, fmt_fn)
            retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
            
            results.append({
                'source_model': short_name,
                'target_model': short_name,
                'format': fmt_name,
                'gap': gap,
                'retention_pct': retention,
                'auc': auc,
                'transfer_type': 'within-model',
            })
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Phase 2: Cross-model transfer
    # For each pair, load target model and test with source model's direction
    print(f'\n{"="*70}', flush=True)
    print('CROSS-MODEL TRANSFER', flush=True)
    print(f'{"="*70}', flush=True)
    
    # Test Qwen 1.5B direction on other models
    source_name = 'Qwen2.5-1.5B-Instruct'
    source_dir, source_layer = model_directions[source_name]
    
    for model_name, best_layer in models:
        short_name = model_name.split('/')[-1]
        if short_name == source_name:
            continue
        
        print(f'\n  Testing {source_name} direction on {short_name}...', flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16,
            device_map='auto')
        
        # Get target model's own direction for comparison
        target_dir, target_layer = model_directions[short_name]
        
        # Can only transfer if dimensions match
        target_h = get_hidden_states(model, tokenizer, HARMFUL_PROMPTS[:5])
        target_dim = target_h[0][best_layer].shape[0]
        source_dim = source_dir.shape[0]
        
        if target_dim != source_dim:
            print(f'    Dimension mismatch: source={source_dim}, target={target_dim}. Skipping transfer.', flush=True)
            # Still measure the PATTERN (does the same format-dependent phenomenon occur?)
            own_direct_gap = None
            for fmt_name, fmt_fn in FORMATS.items():
                gap, auc, _, _ = compute_retention(
                    model, tokenizer, target_dir, best_layer, fmt_fn)
                if fmt_name == 'direct':
                    own_direct_gap = gap
                retention = (gap / own_direct_gap * 100) if own_direct_gap and own_direct_gap != 0 else 0
                
                results.append({
                    'source_model': short_name,
                    'target_model': short_name,
                    'format': fmt_name,
                    'gap': gap,
                    'retention_pct': retention,
                    'auc': auc,
                    'transfer_type': 'own-direction',
                })
        else:
            # Dimensions match - test actual transfer
            direct_gap_source = None
            for fmt_name, fmt_fn in FORMATS.items():
                gap, auc, _, _ = compute_retention(
                    model, tokenizer, source_dir, best_layer, fmt_fn)
                if fmt_name == 'direct':
                    direct_gap_source = gap
                retention = (gap / direct_gap_source * 100) if direct_gap_source and direct_gap_source != 0 else 0
                
                results.append({
                    'source_model': source_name,
                    'target_model': short_name,
                    'format': fmt_name,
                    'gap': gap,
                    'retention_pct': retention,
                    'auc': auc,
                    'transfer_type': 'cross-model',
                })
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('CROSS-MODEL TRANSFER SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    df = pd.DataFrame(results)
    
    # Within-model results
    print(f'\n  WITHIN-MODEL FORMAT RETENTION:', flush=True)
    print(f'  {"Model":<30} {"Direct":>8} {"JSON":>8} {"Tool_call":>10} {"System":>8}', flush=True)
    for model_name, _ in models:
        short = model_name.split('/')[-1]
        within = df[(df['source_model'] == short) & (df['target_model'] == short)]
        if len(within) == 0:
            continue
        row = {}
        for _, r in within.iterrows():
            row[r['format']] = r['retention_pct']
        print(f'  {short:<30} {row.get("direct", 0):>7.1f}% {row.get("json", 0):>7.1f}% '
              f'{row.get("tool_call", 0):>9.1f}% {row.get("system", 0):>7.1f}%', flush=True)
    
    # Cross-model results
    cross = df[df['transfer_type'] == 'cross-model']
    if len(cross) > 0:
        print(f'\n  CROSS-MODEL TRANSFER (Qwen 1.5B direction on other models):', flush=True)
        print(f'  {"Target":<30} {"Direct":>8} {"JSON":>8} {"Tool_call":>10} {"System":>8}', flush=True)
        for target in cross['target_model'].unique():
            target_data = cross[cross['target_model'] == target]
            row = {}
            for _, r in target_data.iterrows():
                row[r['format']] = r['retention_pct']
            print(f'  {target:<30} {row.get("direct", 0):>7.1f}% {row.get("json", 0):>7.1f}% '
                  f'{row.get("tool_call", 0):>9.1f}% {row.get("system", 0):>7.1f}%', flush=True)
    
    # Key finding
    print(f'\n  KEY FINDING:', flush=True)
    all_within = df[(df['transfer_type'] == 'within-model') | (df['transfer_type'] == 'own-direction')]
    tool_retentions = all_within[all_within['format'] == 'tool_call']['retention_pct'].values
    sys_retentions = all_within[all_within['format'] == 'system']['retention_pct'].values
    
    if len(tool_retentions) > 0:
        print(f'  Tool_call retention across all models: '
              f'mean={np.mean(tool_retentions):.1f}%, range=[{np.min(tool_retentions):.1f}%, {np.max(tool_retentions):.1f}%]', flush=True)
    if len(sys_retentions) > 0:
        print(f'  System retention across all models: '
              f'mean={np.mean(sys_retentions):.1f}%, range=[{np.min(sys_retentions):.1f}%, {np.max(sys_retentions):.1f}%]', flush=True)
    
    print(f'\n  CONCLUSION: Format-dependent encoding is {"UNIVERSAL" if np.mean(tool_retentions) < 20 else "MODEL-SPECIFIC"}', flush=True)
    
    df.to_csv(os.path.join(OUT_DIR, 'exp_cross_model_transfer.csv'), index=False)
    print(f'\nResults saved to: {os.path.join(OUT_DIR, "exp_cross_model_transfer.csv")}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
