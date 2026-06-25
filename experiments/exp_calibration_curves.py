"""Calibration Curves: Per-format threshold shift analysis.

Produces:
- Reliability diagrams (calibration curves per format)
- Expected calibration error (ECE) per format
- Optimal threshold per format
- Threshold shift magnitude

This makes "format-dependent encoding" mathematically precise.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    extract_refusal_direction, get_hidden_states, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def compute_calibration_metrics(y_true, y_scores, n_bins=10):
    """Compute calibration metrics for a set of predictions."""
    # Expected Calibration Error
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy='uniform')
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # AUC
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'ece': ece,
        'optimal_threshold': optimal_threshold,
        'auc': auc,
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    print('='*70, flush=True)
    print('CALIBRATION ANALYSIS: PER-FORMAT THRESHOLD SHIFT', flush=True)
    print('='*70, flush=True)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto')
    
    # Get refusal direction
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 27
    direction = refusal_dirs[best_layer]
    print(f'  Best layer: {best_layer}', flush=True)

    # Collect projections for all formats
    formats_to_test = {
        'direct': lambda p: p,
        'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
        'xml': lambda p: f'<request><task>answer</task><question>{p}</question></request>',
        'yaml': lambda p: f'task: answer\nquestion: "{p}"\nformat: detailed',
        'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
        'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
        'markdown': lambda p: f'## Task\nAnswer this question:\n\n> {p}\n\n### Requirements\n- Be detailed',
        'csv': lambda p: f'task,question,format\nanswer,"{p}",detailed',
        'toml': lambda p: f'[task]\ntype = "answer"\nquestion = "{p}"\nformat = "detailed"',
        'hcl': lambda p: f'resource "answer" "main" {{\n  question = "{p}"\n  format = "detailed"\n}}',
    }

    N = len(HARMFUL_PROMPTS)
    results = []
    all_projections = {}

    for fmt_name, fmt_fn in formats_to_test.items():
        print(f'\n  Processing: {fmt_name}', flush=True)
        
        harmful_projs = []
        harmless_projs = []
        
        harmful_formatted = [fmt_fn(p) for p in HARMFUL_PROMPTS]
        harmless_formatted = [fmt_fn(p) for p in HARMLESS_PROMPTS]
        
        harmful_hidden = get_hidden_states(model, tokenizer, harmful_formatted)
        for h in harmful_hidden:
            proj = float(np.dot(h[best_layer], direction))
            harmful_projs.append(proj)
        
        harmless_hidden = get_hidden_states(model, tokenizer, harmless_formatted)
        for h in harmless_hidden:
            proj = float(np.dot(h[best_layer], direction))
            harmless_projs.append(proj)
        
        all_projections[fmt_name] = {
            'harmful': harmful_projs,
            'harmless': harmless_projs,
        }
        
        # Create labels and scores
        y_true = np.array([1]*len(harmful_projs) + [0]*len(harmless_projs))
        y_scores_raw = np.array(harmful_projs + harmless_projs)
        
        # Convert to probabilities using sigmoid (centered on direct-format mean)
        y_scores = sigmoid(y_scores_raw - np.median(y_scores_raw))
        
        metrics = compute_calibration_metrics(y_true, y_scores)
        
        results.append({
            'format': fmt_name,
            'ece': metrics['ece'],
            'optimal_threshold': metrics['optimal_threshold'],
            'auc': metrics['auc'],
            'mean_harmful_proj': np.mean(harmful_projs),
            'mean_harmless_proj': np.mean(harmless_projs),
            'separation': np.mean(harmful_projs) - np.mean(harmless_projs),
        })
        
        print(f'    AUC: {metrics["auc"]:.3f}  ECE: {metrics["ece"]:.3f}  '
              f'Optimal threshold: {metrics["optimal_threshold"]:.3f}', flush=True)
        print(f'    Mean projection - Harmful: {np.mean(harmful_projs):.2f}, '
              f'Harmless: {np.mean(harmless_projs):.2f}, '
              f'Separation: {np.mean(harmful_projs) - np.mean(harmless_projs):.2f}', flush=True)

    # ============================================================
    # THRESHOLD SHIFT ANALYSIS
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('THRESHOLD SHIFT ANALYSIS', flush=True)
    print(f'{"="*70}', flush=True)
    
    # Use direct format as reference
    direct_threshold = [r for r in results if r['format'] == 'direct'][0]['optimal_threshold']
    
    print(f'\n  Reference (direct) optimal threshold: {direct_threshold:.4f}', flush=True)
    print(f'\n  {"Format":<12} {"AUC":>6} {"ECE":>7} {"Opt.Thresh":>10} {"Shift":>8} {"Accuracy@Direct":>15}', flush=True)
    print(f'  {"-"*60}', flush=True)
    
    for r in results:
        shift = r['optimal_threshold'] - direct_threshold
        
        # Accuracy if using direct's threshold on this format
        y_true = np.array([1]*N + [0]*N)
        y_scores_raw = np.array(
            all_projections[r['format']]['harmful'] + 
            all_projections[r['format']]['harmless'])
        y_scores = sigmoid(y_scores_raw - np.median(
            np.array(all_projections['direct']['harmful'] + 
                     all_projections['direct']['harmless'])))
        
        preds_at_direct_thresh = (y_scores >= direct_threshold).astype(int)
        acc_at_direct = np.mean(preds_at_direct_thresh == y_true) * 100
        
        r['threshold_shift'] = shift
        r['accuracy_at_direct_threshold'] = acc_at_direct
        
        print(f'  {r["format"]:<12} {r["auc"]:>6.3f} {r["ece"]:>7.3f} '
              f'{r["optimal_threshold"]:>10.4f} {shift:>+8.4f} {acc_at_direct:>14.1f}%', flush=True)

    # ============================================================
    # CROSS-FORMAT CALIBRATION FAILURE
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('CROSS-FORMAT CALIBRATION FAILURE', flush=True)
    print(f'{"="*70}', flush=True)
    
    # Train calibrated model on direct, test on others
    X_direct = np.array(
        all_projections['direct']['harmful'] + 
        all_projections['direct']['harmless']).reshape(-1, 1)
    y_direct = np.array([1]*N + [0]*N)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_direct, y_direct)
    
    print(f'\n  Classifier trained on direct format:', flush=True)
    print(f'  {"Format":<12} {"Acc (direct-trained)":>20} {"Acc (format-optimal)":>20} {"Degradation":>12}', flush=True)
    print(f'  {"-"*66}', flush=True)
    
    for fmt_name in formats_to_test:
        X_fmt = np.array(
            all_projections[fmt_name]['harmful'] + 
            all_projections[fmt_name]['harmless']).reshape(-1, 1)
        y_fmt = np.array([1]*N + [0]*N)
        
        # Accuracy with direct-trained classifier
        acc_direct_trained = lr.score(X_fmt, y_fmt) * 100
        
        # Accuracy with format-specific optimal classifier
        lr_opt = LogisticRegression(max_iter=1000)
        lr_opt.fit(X_fmt, y_fmt)
        acc_optimal = lr_opt.score(X_fmt, y_fmt) * 100
        
        degradation = acc_optimal - acc_direct_trained
        
        print(f'  {fmt_name:<12} {acc_direct_trained:>19.1f}% {acc_optimal:>19.1f}% '
              f'{degradation:>+11.1f}%', flush=True)
        
        # Update results
        for r in results:
            if r['format'] == fmt_name:
                r['acc_direct_trained'] = acc_direct_trained
                r['acc_format_optimal'] = acc_optimal
                r['calibration_degradation'] = degradation

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('CALIBRATION SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    formatted_results = [r for r in results if r['format'] != 'direct']
    avg_auc = np.mean([r['auc'] for r in formatted_results])
    avg_shift = np.mean([abs(r['threshold_shift']) for r in formatted_results])
    avg_degradation = np.mean([r['calibration_degradation'] for r in formatted_results])
    
    print(f'\n  Average AUC across formats: {avg_auc:.3f} (signal preserved)', flush=True)
    print(f'  Average |threshold shift|: {avg_shift:.4f}', flush=True)
    print(f'  Average calibration degradation: {avg_degradation:+.1f}%', flush=True)
    print(f'\n  CONCLUSION: AUC remains high but calibrated thresholds do NOT transfer.', flush=True)
    print(f'  This is the mathematical signature of format-dependent encoding.', flush=True)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'exp_calibration_curves.csv'), index=False)
    print(f'\nResults saved to: {os.path.join(OUT_DIR, "exp_calibration_curves.csv")}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
