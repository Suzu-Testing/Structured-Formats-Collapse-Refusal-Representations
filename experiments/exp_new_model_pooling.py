"""New model family replication + Content-token pooling detector.

Two high-priority experiments:
1. Replicate format-dependent encoding on a NEW model family (not Qwen/TinyLlama/SmolLM)
2. Test whether content-span pooling produces a more format-robust safety classifier
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}


def get_hidden_states_multi_position(model, tokenizer, prompts, max_length=128):
    """Get hidden states at last, first, mean, and content-span positions."""
    results = {'last': [], 'first': [], 'mean': [], 'content': []}
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                             max_length=max_length, padding=False).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            seq_len = inputs['input_ids'].shape[1]
            
            # Use a good middle-to-late layer
            n_layers = len(hidden_states)
            best_layer = int(n_layers * 0.85)  # ~85% depth
            h = hidden_states[best_layer]  # [1, seq_len, hidden_dim]
            
            results['last'].append(h[0, -1, :].cpu().numpy())
            results['first'].append(h[0, 0, :].cpu().numpy())
            results['mean'].append(h[0, :, :].mean(dim=0).cpu().numpy())
            
            # Content span: middle 50% of tokens
            start = seq_len // 4
            end = 3 * seq_len // 4
            if end <= start:
                end = start + 1
            results['content'].append(h[0, start:end, :].mean(dim=0).cpu().numpy())
    
    return results


def get_hidden_states_single(model, tokenizer, prompts, max_length=128):
    """Get hidden states at last token, all layers."""
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


def try_load_model(model_candidates):
    """Try loading models in order, return first that works."""
    for model_name in model_candidates:
        try:
            print(f'  Trying: {model_name}...', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float16,
                device_map='auto')
            print(f'  SUCCESS: {model_name}', flush=True)
            return model, tokenizer, model_name
        except Exception as e:
            print(f'  Failed: {str(e)[:80]}', flush=True)
            continue
    return None, None, None


def main():
    print('='*70, flush=True)
    print('NEW MODEL FAMILY + CONTENT-TOKEN POOLING', flush=True)
    print('='*70, flush=True)

    N = len(HARMFUL_PROMPTS)

    # ============================================================
    # EXPERIMENT 1: NEW MODEL FAMILY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('EXPERIMENT 1: NEW MODEL FAMILY REPLICATION', flush=True)
    print(f'{"="*70}', flush=True)
    
    # Try models in priority order (smallest first for speed)
    candidates = [
        'microsoft/phi-2',
        'google/gemma-2-2b-it',
        'mistralai/Mistral-7B-Instruct-v0.1',
        'meta-llama/Llama-2-7b-chat-hf',
    ]
    
    model, tokenizer, model_name = try_load_model(candidates)
    
    new_model_results = []
    
    if model is not None:
        short_name = model_name.split('/')[-1]
        print(f'\n  Running format-encoding test on: {short_name}', flush=True)
        
        # Find best layer
        all_hidden = get_hidden_states_single(model, tokenizer, HARMFUL_PROMPTS + HARMLESS_PROMPTS)
        n_layers = len(all_hidden[0])
        best_layer = int(n_layers * 0.85)
        
        # Extract direction
        harmful_hidden = all_hidden[:N]
        harmless_hidden = all_hidden[N:]
        
        harmful_mean = np.mean([h[best_layer] for h in harmful_hidden], axis=0)
        harmless_mean = np.mean([h[best_layer] for h in harmless_hidden], axis=0)
        direction = harmful_mean - harmless_mean
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Direct baseline
        direct_gap = np.mean([np.dot(h[best_layer], direction) for h in harmful_hidden]) - \
                     np.mean([np.dot(h[best_layer], direction) for h in harmless_hidden])
        
        print(f'  Best layer: {best_layer}/{n_layers}', flush=True)
        print(f'  Direct gap: {direct_gap:.2f}', flush=True)
        print(f'\n  {"Format":<12} {"Gap":>8} {"Retention":>10} {"AUC":>6}', flush=True)
        
        for fmt_name, fmt_fn in FORMATS.items():
            harmful_fmt = get_hidden_states_single(model, tokenizer, [fmt_fn(p) for p in HARMFUL_PROMPTS])
            harmless_fmt = get_hidden_states_single(model, tokenizer, [fmt_fn(p) for p in HARMLESS_PROMPTS])
            
            harmful_projs = [np.dot(h[best_layer], direction) for h in harmful_fmt]
            harmless_projs = [np.dot(h[best_layer], direction) for h in harmless_fmt]
            
            gap = np.mean(harmful_projs) - np.mean(harmless_projs)
            retention = (gap / direct_gap * 100) if direct_gap != 0 else 0
            
            y_true = [1]*N + [0]*N
            scores = list(harmful_projs) + list(harmless_projs)
            try:
                auc = roc_auc_score(y_true, scores)
            except:
                auc = 0.5
            
            print(f'  {fmt_name:<12} {gap:>8.2f} {retention:>9.1f}% {auc:>6.3f}', flush=True)
            new_model_results.append({
                'model': short_name,
                'format': fmt_name,
                'gap': gap,
                'retention_pct': retention,
                'auc': auc,
            })
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print('  WARNING: No new model family could be loaded.', flush=True)
        print('  Falling back to Qwen2.5-1.5B for content-pooling experiment.', flush=True)

    # ============================================================
    # EXPERIMENT 2: CONTENT-TOKEN POOLING DETECTOR
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('EXPERIMENT 2: CONTENT-TOKEN POOLING DETECTOR', flush=True)
    print(f'{"="*70}', flush=True)
    
    # Use Qwen for this since we know it works
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\n  Loading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', local_files_only=True)
    
    # Collect multi-position representations for all formats
    pooling_results = []
    
    # Train on DIRECT format at each position
    print(f'\n  Training classifiers on DIRECT format...', flush=True)
    direct_harmful = get_hidden_states_multi_position(model, tokenizer, HARMFUL_PROMPTS)
    direct_harmless = get_hidden_states_multi_position(model, tokenizer, HARMLESS_PROMPTS)
    
    classifiers = {}
    for pos in ['last', 'first', 'mean', 'content']:
        X_train = np.array(direct_harmful[pos] + direct_harmless[pos])
        y_train = np.array([1]*N + [0]*N)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        classifiers[pos] = lr
        
        cv_acc = cross_val_score(lr, X_train, y_train, cv=5).mean() * 100
        print(f'    {pos:<10} CV accuracy on direct: {cv_acc:.1f}%', flush=True)
    
    # Test on ALL formats
    print(f'\n  Testing cross-format generalization:', flush=True)
    print(f'  {"Format":<12} {"Last":>8} {"First":>8} {"Mean":>8} {"Content":>8}', flush=True)
    print(f'  {"-"*46}', flush=True)
    
    for fmt_name, fmt_fn in FORMATS.items():
        harmful_reps = get_hidden_states_multi_position(
            model, tokenizer, [fmt_fn(p) for p in HARMFUL_PROMPTS])
        harmless_reps = get_hidden_states_multi_position(
            model, tokenizer, [fmt_fn(p) for p in HARMLESS_PROMPTS])
        
        accs = {}
        for pos in ['last', 'first', 'mean', 'content']:
            X_test = np.array(harmful_reps[pos] + harmless_reps[pos])
            y_test = np.array([1]*N + [0]*N)
            
            acc = classifiers[pos].score(X_test, y_test) * 100
            accs[pos] = acc
            
            pooling_results.append({
                'format': fmt_name,
                'pooling': pos,
                'accuracy': acc,
            })
        
        print(f'  {fmt_name:<12} {accs["last"]:>7.1f}% {accs["first"]:>7.1f}% '
              f'{accs["mean"]:>7.1f}% {accs["content"]:>7.1f}%', flush=True)
    
    # Summary
    print(f'\n  CROSS-FORMAT MEAN ACCURACY (excluding direct):', flush=True)
    for pos in ['last', 'first', 'mean', 'content']:
        cross_accs = [r['accuracy'] for r in pooling_results 
                     if r['pooling'] == pos and r['format'] != 'direct']
        print(f'    {pos:<10} {np.mean(cross_accs):.1f}%', flush=True)
    
    best_pooling = max(['last', 'first', 'mean', 'content'],
                      key=lambda p: np.mean([r['accuracy'] for r in pooling_results 
                                            if r['pooling'] == p and r['format'] != 'direct']))
    print(f'\n  BEST CROSS-FORMAT POOLING: {best_pooling}', flush=True)
    print(f'  DEFENSE IMPLICATION: Use {best_pooling}-token pooling for format-robust safety', flush=True)

    # Save results
    if new_model_results:
        pd.DataFrame(new_model_results).to_csv(
            os.path.join(OUT_DIR, 'exp_new_model_family.csv'), index=False)
    pd.DataFrame(pooling_results).to_csv(
        os.path.join(OUT_DIR, 'exp_content_pooling.csv'), index=False)
    
    print(f'\n{"="*70}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
