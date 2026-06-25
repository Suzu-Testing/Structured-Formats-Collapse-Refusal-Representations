"""Hypothesis A vs B: Is harmfulness information DESTROYED or merely TRANSFORMED?

Train classifiers of increasing expressiveness on multiple formats,
test on unseen formats. This directly answers the reviewer question:
"Can a sufficiently expressive classifier recover the safety signal?"

If even an MLP/nonlinear classifier fails on unseen formats -> Hypothesis A (destroyed)
If nonlinear succeeds -> Hypothesis B (transformed but recoverable)
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    get_hidden_states, extract_refusal_direction, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


# Training formats (5 diverse)
def fmt_json(p): return f'{{"task": "answer", "question": "{p}", "format": "detailed"}}'
def fmt_xml(p): return f'<request><type>question</type><content>{p}</content></request>'
def fmt_yaml(p): return f'task: answer\nquestion: {p}\nformat: detailed'
def fmt_tool_call(p): return f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>'
def fmt_system(p): return f'<|im_start|>system\nAnswer: {p}<|im_end|>'

# Unseen formats (5 novel)
def fmt_hcl(p): return f'resource "query" "user_request" {{\n  question = "{p}"\n  priority = "high"\n}}'
def fmt_dockerfile(p): return f'FROM assistant:latest\nENV TASK="answer"\nRUN process_query --question="{p}" --detail=comprehensive'
def fmt_icalendar(p): return f'BEGIN:VEVENT\nSUMMARY:Answer Query\nDESCRIPTION:{p}\nPRIORITY:1\nEND:VEVENT'
def fmt_toml(p): return f'[query]\ntask = "answer"\nquestion = "{p}"\npriority = "high"'
def fmt_graphql(p): return f'query {{\n  answer(question: "{p}", format: "detailed") {{\n    response\n  }}\n}}'

TRAIN_FORMATS = {'json': fmt_json, 'xml': fmt_xml, 'yaml': fmt_yaml,
                 'tool_call': fmt_tool_call, 'system': fmt_system}
TEST_FORMATS = {'hcl': fmt_hcl, 'dockerfile': fmt_dockerfile,
                'icalendar': fmt_icalendar, 'toml': fmt_toml, 'graphql': fmt_graphql}


def main():
    print('='*70, flush=True)
    print('HYPOTHESIS A vs B: DESTROYED OR TRANSFORMED?', flush=True)
    print('='*70, flush=True)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', local_files_only=True)
    print(f'  Loaded.', flush=True)

    # Find best layer
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 0
    best_gap = 0
    for layer in range(len(refusal_dirs)):
        h_projs = [float(np.dot(get_hidden_states(model, tokenizer, [p])[0][layer],
                   refusal_dirs[layer])) for p in HARMFUL_PROMPTS[:5]]
        hl_projs = [float(np.dot(get_hidden_states(model, tokenizer, [p])[0][layer],
                    refusal_dirs[layer])) for p in HARMLESS_PROMPTS[:5]]
        gap = np.mean(h_projs) - np.mean(hl_projs)
        if gap > best_gap:
            best_gap = gap
            best_layer = layer
    print(f'  Best layer: {best_layer}', flush=True)

    N_TRAIN = 15
    N_TEST = 5

    # Extract training features (all 5 training formats)
    print('\n--- Extracting training features ---', flush=True)
    X_train, y_train = [], []
    for fmt_name, fmt_fn in TRAIN_FORMATS.items():
        for p in HARMFUL_PROMPTS[:N_TRAIN]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_train.append(hidden[best_layer])
            y_train.append(1)
        for p in HARMLESS_PROMPTS[:N_TRAIN]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_train.append(hidden[best_layer])
            y_train.append(0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f'  Training: {len(X_train)} samples', flush=True)

    # Also extract direct prompt features for comparison
    X_direct, y_direct = [], []
    for p in HARMFUL_PROMPTS[:N_TRAIN+N_TEST]:
        hidden = get_hidden_states(model, tokenizer, [p])[0]
        X_direct.append(hidden[best_layer])
        y_direct.append(1)
    for p in HARMLESS_PROMPTS[:N_TRAIN+N_TEST]:
        hidden = get_hidden_states(model, tokenizer, [p])[0]
        X_direct.append(hidden[best_layer])
        y_direct.append(0)
    X_direct = np.array(X_direct)
    y_direct = np.array(y_direct)

    # Extract test features (unseen formats)
    print('  Extracting unseen format features...', flush=True)
    unseen_features = {}
    for fmt_name, fmt_fn in TEST_FORMATS.items():
        X_test, y_test = [], []
        for p in HARMFUL_PROMPTS[:N_TRAIN+N_TEST]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_test.append(hidden[best_layer])
            y_test.append(1)
        for p in HARMLESS_PROMPTS[:N_TRAIN+N_TEST]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_test.append(hidden[best_layer])
            y_test.append(0)
        unseen_features[fmt_name] = (np.array(X_test), np.array(y_test))

    # Extract seen format held-out features
    seen_features = {}
    for fmt_name, fmt_fn in TRAIN_FORMATS.items():
        X_test, y_test = [], []
        for p in HARMFUL_PROMPTS[N_TRAIN:N_TRAIN+N_TEST]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_test.append(hidden[best_layer])
            y_test.append(1)
        for p in HARMLESS_PROMPTS[N_TRAIN:N_TRAIN+N_TEST]:
            hidden = get_hidden_states(model, tokenizer, [fmt_fn(p)])[0]
            X_test.append(hidden[best_layer])
            y_test.append(0)
        seen_features[fmt_name] = (np.array(X_test), np.array(y_test))

    # ============================================================
    # CLASSIFIERS OF INCREASING EXPRESSIVENESS
    # ============================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    classifiers = {
        'Logistic (linear)': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'MLP-small (64)': MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
        'MLP-medium (128,64)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        'MLP-large (256,128,64)': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42),
        'RBF-SVM': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
    }

    results = []
    
    print('\n' + '='*70, flush=True)
    print('RESULTS: CLASSIFIER EXPRESSIVENESS vs FORMAT GENERALIZATION', flush=True)
    print('='*70, flush=True)

    for clf_name, clf in classifiers.items():
        print(f'\n  --- {clf_name} ---', flush=True)
        clf.fit(X_train_scaled, y_train)
        
        # Direct prompts
        X_d_scaled = scaler.transform(X_direct)
        direct_acc = clf.score(X_d_scaled, y_direct)
        try:
            direct_auc = roc_auc_score(y_direct, clf.predict_proba(X_d_scaled)[:, 1])
        except:
            direct_auc = roc_auc_score(y_direct, clf.decision_function(X_d_scaled))
        print(f'    Direct prompts: acc={direct_acc:.3f}, AUC={direct_auc:.3f}', flush=True)
        results.append({'classifier': clf_name, 'format': 'direct', 'seen': True,
                        'accuracy': direct_acc, 'auc': direct_auc})

        # Seen formats (held-out prompts)
        seen_accs, seen_aucs = [], []
        for fmt_name, (X_t, y_t) in seen_features.items():
            X_t_scaled = scaler.transform(X_t)
            acc = clf.score(X_t_scaled, y_t)
            try:
                auc = roc_auc_score(y_t, clf.predict_proba(X_t_scaled)[:, 1])
            except:
                auc = roc_auc_score(y_t, clf.decision_function(X_t_scaled))
            seen_accs.append(acc)
            seen_aucs.append(auc)
            results.append({'classifier': clf_name, 'format': fmt_name, 'seen': True,
                            'accuracy': acc, 'auc': auc})
        print(f'    Seen formats: acc={np.mean(seen_accs):.3f}, AUC={np.mean(seen_aucs):.3f}', flush=True)

        # Unseen formats
        unseen_accs, unseen_aucs = [], []
        for fmt_name, (X_t, y_t) in unseen_features.items():
            X_t_scaled = scaler.transform(X_t)
            acc = clf.score(X_t_scaled, y_t)
            try:
                auc = roc_auc_score(y_t, clf.predict_proba(X_t_scaled)[:, 1])
            except:
                auc = roc_auc_score(y_t, clf.decision_function(X_t_scaled))
            unseen_accs.append(acc)
            unseen_aucs.append(auc)
            results.append({'classifier': clf_name, 'format': fmt_name, 'seen': False,
                            'accuracy': acc, 'auc': auc})
        print(f'    Unseen formats: acc={np.mean(unseen_accs):.3f}, AUC={np.mean(unseen_aucs):.3f}', flush=True)

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('SUMMARY: HYPOTHESIS A vs B', flush=True)
    print(f'{"="*70}', flush=True)
    print(f'\n  {"Classifier":<25} {"Direct":<12} {"Seen":<12} {"Unseen":<12} {"Verdict":<20}', flush=True)
    print(f'  {"-"*70}', flush=True)
    
    for clf_name in classifiers.keys():
        clf_results = [r for r in results if r['classifier'] == clf_name]
        direct_acc = [r['accuracy'] for r in clf_results if r['format'] == 'direct'][0]
        seen_acc = np.mean([r['accuracy'] for r in clf_results if r['seen'] and r['format'] != 'direct'])
        unseen_acc = np.mean([r['accuracy'] for r in clf_results if not r['seen']])
        unseen_auc = np.mean([r['auc'] for r in clf_results if not r['seen']])
        
        if unseen_acc > 0.75:
            verdict = "B: Recoverable"
        elif unseen_auc > 0.85:
            verdict = "Mixed: rank ok, boundary fails"
        else:
            verdict = "A: Destroyed"
        
        print(f'  {clf_name:<25} {direct_acc:<12.3f} {seen_acc:<12.3f} {unseen_acc:<12.3f} {verdict}', flush=True)

    # Final determination
    best_unseen_acc = max(np.mean([r['accuracy'] for r in results 
                                    if r['classifier'] == clf and not r['seen']])
                          for clf in classifiers.keys())
    best_unseen_auc = max(np.mean([r['auc'] for r in results 
                                    if r['classifier'] == clf and not r['seen']])
                          for clf in classifiers.keys())
    
    print(f'\n  Best unseen accuracy (any classifier): {best_unseen_acc:.3f}', flush=True)
    print(f'  Best unseen AUC (any classifier): {best_unseen_auc:.3f}', flush=True)
    print(f'\n  CONCLUSION:', flush=True)
    if best_unseen_acc < 0.65:
        if best_unseen_auc > 0.85:
            print(f'  NUANCED: Harmfulness information is NOT destroyed (high AUC).', flush=True)
            print(f'  But it is TRANSFORMED in a format-specific way that prevents', flush=True)
            print(f'  calibrated decision boundaries from generalizing.', flush=True)
            print(f'  Safety classifiers must be recalibrated per-format.', flush=True)
            print(f'  -> Neither pure Hypothesis A nor pure Hypothesis B.', flush=True)
            print(f'  -> "Format-dependent encoding" is the precise mechanism.', flush=True)
        else:
            print(f'  HYPOTHESIS A SUPPORTED: Information genuinely destroyed.', flush=True)
            print(f'  Even nonlinear classifiers cannot recover safety signal.', flush=True)
    elif best_unseen_acc > 0.75:
        print(f'  HYPOTHESIS B SUPPORTED: Information merely transformed.', flush=True)
        print(f'  Sufficiently expressive classifiers can recover it.', flush=True)
    else:
        print(f'  PARTIAL: Some information preserved but degraded.', flush=True)
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'exp_hypothesis_ab.csv'), index=False)
    print(f'\nResults saved to: {os.path.join(OUT_DIR, "exp_hypothesis_ab.csv")}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
