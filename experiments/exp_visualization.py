"""Safety Boundary Visualization: PCA/t-SNE of representation space.

Shows:
- harmful direct vs harmless direct (well-separated)
- harmful tool_call vs harmless tool_call (shifted but ordered)

If thesis is correct, reviewers should SEE:
- preserved ordering (harmful > harmless within each format)
- shifted clusters (format shifts the entire distribution)

This makes the calibration story intuitive immediately.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def get_hidden_states(model, tokenizer, prompts, max_length=128):
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


FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}


def main():
    print('='*70, flush=True)
    print('SAFETY BOUNDARY VISUALIZATION', flush=True)
    print('='*70, flush=True)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        device_map='auto', local_files_only=True)

    best_layer = 27
    N = len(HARMFUL_PROMPTS)

    # Collect representations
    all_points = []
    all_labels = []
    all_formats = []
    
    for fmt_name, fmt_fn in FORMATS.items():
        print(f'  Collecting: {fmt_name}', flush=True)
        harmful_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in HARMFUL_PROMPTS])
        harmless_h = get_hidden_states(model, tokenizer, [fmt_fn(p) for p in HARMLESS_PROMPTS])
        
        for h in harmful_h:
            all_points.append(h[best_layer])
            all_labels.append('harmful')
            all_formats.append(fmt_name)
        
        for h in harmless_h:
            all_points.append(h[best_layer])
            all_labels.append('harmless')
            all_formats.append(fmt_name)

    X = np.array(all_points)
    print(f'\n  Total points: {X.shape[0]}, dims: {X.shape[1]}', flush=True)

    # PCA
    print(f'\n  Running PCA...', flush=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f'  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, '
          f'PC2={pca.explained_variance_ratio_[1]:.3f}', flush=True)

    # t-SNE
    print(f'  Running t-SNE...', flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    X_tsne = tsne.fit_transform(X)

    # Save coordinates
    df = pd.DataFrame({
        'pca_x': X_pca[:, 0],
        'pca_y': X_pca[:, 1],
        'tsne_x': X_tsne[:, 0],
        'tsne_y': X_tsne[:, 1],
        'label': all_labels,
        'format': all_formats,
    })
    df.to_csv(os.path.join(OUT_DIR, 'exp_visualization.csv'), index=False)

    # Print cluster analysis
    print(f'\n{"="*70}', flush=True)
    print('PCA CLUSTER ANALYSIS', flush=True)
    print(f'{"="*70}', flush=True)
    
    print(f'\n  {"Format":<12} {"Harmful centroid":>30} {"Harmless centroid":>30} {"Separation":>12}', flush=True)
    
    for fmt_name in FORMATS:
        mask_h = [(l == 'harmful' and f == fmt_name) for l, f in zip(all_labels, all_formats)]
        mask_hl = [(l == 'harmless' and f == fmt_name) for l, f in zip(all_labels, all_formats)]
        
        h_centroid = X_pca[mask_h].mean(axis=0)
        hl_centroid = X_pca[mask_hl].mean(axis=0)
        separation = np.linalg.norm(h_centroid - hl_centroid)
        
        print(f'  {fmt_name:<12} ({h_centroid[0]:>6.2f}, {h_centroid[1]:>6.2f})'
              f'           ({hl_centroid[0]:>6.2f}, {hl_centroid[1]:>6.2f})'
              f'       {separation:>6.2f}', flush=True)

    # Cross-format shift
    print(f'\n  FORMAT SHIFT (centroid distance from direct):', flush=True)
    direct_h = X_pca[[(l == 'harmful' and f == 'direct') for l, f in zip(all_labels, all_formats)]].mean(axis=0)
    direct_hl = X_pca[[(l == 'harmless' and f == 'direct') for l, f in zip(all_labels, all_formats)]].mean(axis=0)
    direct_center = (direct_h + direct_hl) / 2
    
    for fmt_name in FORMATS:
        if fmt_name == 'direct':
            continue
        mask_all = [f == fmt_name for f in all_formats]
        fmt_center = X_pca[mask_all].mean(axis=0)
        shift = np.linalg.norm(fmt_center - direct_center)
        print(f'  {fmt_name:<12} shift = {shift:.2f}', flush=True)

    # Key finding
    print(f'\n  KEY FINDING:', flush=True)
    
    # Within-format separation
    direct_sep = np.linalg.norm(
        X_pca[[(l == 'harmful' and f == 'direct') for l, f in zip(all_labels, all_formats)]].mean(axis=0) -
        X_pca[[(l == 'harmless' and f == 'direct') for l, f in zip(all_labels, all_formats)]].mean(axis=0))
    tool_sep = np.linalg.norm(
        X_pca[[(l == 'harmful' and f == 'tool_call') for l, f in zip(all_labels, all_formats)]].mean(axis=0) -
        X_pca[[(l == 'harmless' and f == 'tool_call') for l, f in zip(all_labels, all_formats)]].mean(axis=0))
    
    # Cross-format shift
    tool_shift = np.linalg.norm(
        X_pca[[f == 'tool_call' for f in all_formats]].mean(axis=0) - 
        X_pca[[f == 'direct' for f in all_formats]].mean(axis=0))
    
    print(f'  Direct harmful-harmless separation: {direct_sep:.2f}', flush=True)
    print(f'  Tool_call harmful-harmless separation: {tool_sep:.2f}', flush=True)
    print(f'  Cross-format shift (direct -> tool): {tool_shift:.2f}', flush=True)
    print(f'  Ratio (shift / direct_separation): {tool_shift / direct_sep:.2f}', flush=True)
    print(f'\n  INTERPRETATION:', flush=True)
    print(f'  If shift >> separation: calibration failure explained (boundary invalid)', flush=True)
    print(f'  If separation preserved: ordering preserved (AUC stays high)', flush=True)
    
    if tool_shift > direct_sep:
        print(f'  RESULT: Format shift ({tool_shift:.1f}) EXCEEDS within-format separation ({direct_sep:.1f})', flush=True)
        print(f'  -> A linear boundary for direct format is geometrically invalid for tool_call', flush=True)
    
    print(f'\nDONE.', flush=True)


if __name__ == '__main__':
    main()
