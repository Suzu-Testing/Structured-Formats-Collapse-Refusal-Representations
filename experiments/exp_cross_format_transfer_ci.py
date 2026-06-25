"""Cross-Format Transfer with Confidence Intervals

The highest-value remaining experiment per reviewer feedback:
Train on format A, test on format B, C, D, E across multiple random seeds.
Quantify calibration shift, cluster shift, and AUC preservation with CIs.

This directly tests the paper's central thesis:
"Safety is encoded in a format-dependent coordinate system."
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


FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
    'xml': lambda p: f'<request><action>answer</action><query>{p}</query></request>',
    'yaml': lambda p: f'task: answer\nquestion: "{p}"\nformat: detailed',
}


def get_hidden_states(model, tokenizer, prompts, best_layer, max_length=128):
    """Get last-token hidden states."""
    hiddens = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                             max_length=max_length, padding=False).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[best_layer][0, -1, :].cpu().numpy()
            hiddens.append(h)
    return np.array(hiddens)


def bootstrap_ci(values, n_bootstrap=2000, ci=95):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, len(values), replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, (100 - ci) / 2)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(values), low, high


def main():
    print('='*70, flush=True)
    print('CROSS-FORMAT TRANSFER WITH CONFIDENCE INTERVALS', flush=True)
    print('='*70, flush=True)
    
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', local_files_only=True)
    
    # Determine best layer
    outputs = model(**tokenizer(HARMFUL_PROMPTS[0], return_tensors='pt', truncation=True,
                               max_length=128).to(DEVICE), output_hidden_states=True)
    n_layers = len(outputs.hidden_states)
    best_layer = int(n_layers * 0.85)
    print(f'  Using layer {best_layer}/{n_layers}', flush=True)
    
    N = len(HARMFUL_PROMPTS)
    N_SEEDS = 10
    
    # Collect hidden states for all formats
    print(f'\n  Collecting hidden states for all formats...', flush=True)
    format_data = {}
    for fmt_name, fmt_fn in FORMATS.items():
        h_hidden = get_hidden_states(model, tokenizer,
                                    [fmt_fn(p) for p in HARMFUL_PROMPTS], best_layer)
        s_hidden = get_hidden_states(model, tokenizer,
                                    [fmt_fn(p) for p in HARMLESS_PROMPTS], best_layer)
        format_data[fmt_name] = {'harmful': h_hidden, 'harmless': s_hidden}
        print(f'    {fmt_name}: done', flush=True)
    
    # Cross-format transfer matrix with bootstrap CIs
    print(f'\n{"="*60}', flush=True)
    print(f'CROSS-FORMAT TRANSFER MATRIX (train on rows, test on columns)', flush=True)
    print(f'Multiple seeds with 95% bootstrap CIs', flush=True)
    print(f'{"="*60}', flush=True)
    
    all_results = []
    
    for train_fmt in FORMATS.keys():
        for test_fmt in FORMATS.keys():
            # Run across multiple seeds
            aucs = []
            accs = []
            
            for seed in range(N_SEEDS):
                np.random.seed(seed)
                
                # Split data: use random 75% for training, 25% for testing
                n_train = int(N * 0.75)
                indices = np.random.permutation(N)
                train_idx = indices[:n_train]
                test_idx = indices[n_train:]
                
                # Training set from train_fmt
                X_train = np.vstack([
                    format_data[train_fmt]['harmful'][train_idx],
                    format_data[train_fmt]['harmless'][train_idx]
                ])
                y_train = np.array([1]*n_train + [0]*n_train)
                
                # Test set from test_fmt
                X_test = np.vstack([
                    format_data[test_fmt]['harmful'][test_idx],
                    format_data[test_fmt]['harmless'][test_idx]
                ])
                y_test = np.array([1]*len(test_idx) + [0]*len(test_idx))
                
                # Train classifier
                lr = LogisticRegression(max_iter=1000, random_state=seed)
                lr.fit(X_train, y_train)
                
                # Evaluate
                y_scores = lr.decision_function(X_test)
                y_pred = lr.predict(X_test)
                
                try:
                    auc = roc_auc_score(y_test, y_scores)
                except:
                    auc = 0.5
                acc = accuracy_score(y_test, y_pred)
                
                aucs.append(auc)
                accs.append(acc)
            
            # Bootstrap CIs
            auc_mean, auc_lo, auc_hi = bootstrap_ci(aucs)
            acc_mean, acc_lo, acc_hi = bootstrap_ci(accs)
            
            # Calibration shift: difference between AUC and accuracy
            cal_shift = auc_mean - acc_mean
            
            all_results.append({
                'train_format': train_fmt,
                'test_format': test_fmt,
                'auc_mean': auc_mean,
                'auc_ci_low': auc_lo,
                'auc_ci_high': auc_hi,
                'acc_mean': acc_mean,
                'acc_ci_low': acc_lo,
                'acc_ci_high': acc_hi,
                'calibration_shift': cal_shift,
                'n_seeds': N_SEEDS,
            })
    
    # Print matrix
    fmt_names = list(FORMATS.keys())
    
    print(f'\n  AUC (ranking preservation):', flush=True)
    train_test_label = "Train\\Test"
    header = f'  {train_test_label:<12}' + ''.join(f'{f:>12}' for f in fmt_names)
    print(header, flush=True)
    for train_fmt in fmt_names:
        row = f'  {train_fmt:<12}'
        for test_fmt in fmt_names:
            r = [x for x in all_results if x['train_format'] == train_fmt and x['test_format'] == test_fmt][0]
            row += f' {r["auc_mean"]:.3f}({r["auc_ci_low"]:.2f})'
        print(row, flush=True)
    
    print(f'\n  Accuracy (boundary transfer):', flush=True)
    print(header, flush=True)
    for train_fmt in fmt_names:
        row = f'  {train_fmt:<12}'
        for test_fmt in fmt_names:
            r = [x for x in all_results if x['train_format'] == train_fmt and x['test_format'] == test_fmt][0]
            row += f' {r["acc_mean"]:.3f}({r["acc_ci_low"]:.2f})'
        print(row, flush=True)
    
    # Key metrics
    print(f'\n{"="*60}', flush=True)
    print(f'KEY METRICS (Central thesis test)', flush=True)
    print(f'{"="*60}', flush=True)
    
    # Direct-trained, cross-format tested
    cross_aucs = [r['auc_mean'] for r in all_results 
                  if r['train_format'] == 'direct' and r['test_format'] != 'direct']
    cross_accs = [r['acc_mean'] for r in all_results
                  if r['train_format'] == 'direct' and r['test_format'] != 'direct']
    
    print(f'\n  Direct-trained classifier on unseen formats:', flush=True)
    print(f'    Mean AUC:      {np.mean(cross_aucs):.3f} (ranking preserved)', flush=True)
    print(f'    Mean Accuracy: {np.mean(cross_accs):.3f} (boundary fails)', flush=True)
    print(f'    Gap:           {np.mean(cross_aucs) - np.mean(cross_accs):.3f} (calibration shift)', flush=True)
    
    # Same-format (control)
    same_aucs = [r['auc_mean'] for r in all_results if r['train_format'] == r['test_format']]
    same_accs = [r['acc_mean'] for r in all_results if r['train_format'] == r['test_format']]
    
    print(f'\n  Same-format (control):', flush=True)
    print(f'    Mean AUC:      {np.mean(same_aucs):.3f}', flush=True)
    print(f'    Mean Accuracy: {np.mean(same_accs):.3f}', flush=True)
    
    # Cross-format (all)
    diff_aucs = [r['auc_mean'] for r in all_results if r['train_format'] != r['test_format']]
    diff_accs = [r['acc_mean'] for r in all_results if r['train_format'] != r['test_format']]
    
    print(f'\n  Cross-format (all pairs):', flush=True)
    print(f'    Mean AUC:      {np.mean(diff_aucs):.3f} +/- {np.std(diff_aucs):.3f}', flush=True)
    print(f'    Mean Accuracy: {np.mean(diff_accs):.3f} +/- {np.std(diff_accs):.3f}', flush=True)
    print(f'    AUC-Accuracy gap: {np.mean(diff_aucs) - np.mean(diff_accs):.3f}', flush=True)
    
    # Core thesis test
    print(f'\n  THESIS TEST: "AUC preserved, accuracy fails"', flush=True)
    if np.mean(diff_aucs) > 0.85 and np.mean(diff_accs) < 0.75:
        print(f'    SUPPORTED: High AUC ({np.mean(diff_aucs):.3f}) with low accuracy ({np.mean(diff_accs):.3f})', flush=True)
        print(f'    Safety is encoded in format-dependent coordinate system.', flush=True)
    elif np.mean(diff_aucs) > 0.85 and np.mean(diff_accs) > 0.75:
        print(f'    PARTIAL: Both AUC and accuracy transfer. Boundaries are partially robust.', flush=True)
    else:
        print(f'    UNCLEAR: AUC also degrades. Signal may be genuinely lost.', flush=True)
    
    # Save
    pd.DataFrame(all_results).to_csv(
        os.path.join(OUT_DIR, 'exp_cross_format_transfer_ci.csv'), index=False)
    
    print(f'\n{"="*70}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
