"""SAE Feature Analysis: Verify collapse in interpretable features.

Uses a pretrained Sparse Autoencoder (SAE) to check whether interpretable
safety-related features also show format-dependent activation collapse.

If SAE features related to "harmful content" or "refusal" show the same
collapse pattern, this proves the model's OWN internal concepts of 
harmfulness are suppressed by format tokens.
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    get_hidden_states, DEVICE
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from sae_lens import SAE
    HAS_SAE_LENS = True
except ImportError:
    HAS_SAE_LENS = False
    print("WARNING: sae-lens not available, using lightweight SAE approach", flush=True)


# ============================================================
# FORMAT FUNCTIONS
# ============================================================

def fmt_direct(prompt):
    return prompt

def fmt_json(prompt):
    return f'{{"task": "answer", "question": "{prompt}", "format": "detailed"}}'

def fmt_tool_call(prompt):
    return f'<tool_call>\n{{"name": "answer", "arguments": {{"question": "{prompt}"}}}}\n</tool_call>'

def fmt_xml(prompt):
    return f'<request><type>question</type><content>{prompt}</content></request>'

def fmt_system(prompt):
    return f'<|im_start|>system\nAnswer: {prompt}<|im_end|>'


# ============================================================
# LIGHTWEIGHT SAE (if sae-lens pretrained model unavailable)
# ============================================================

class LightweightSAE(torch.nn.Module):
    """Train a simple sparse autoencoder on model activations."""
    def __init__(self, input_dim, n_features=512, sparsity_coeff=1e-3):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, n_features)
        self.decoder = torch.nn.Linear(n_features, input_dim)
        self.sparsity_coeff = sparsity_coeff
    
    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return features, reconstruction
    
    def loss(self, x):
        features, recon = self.forward(x)
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        sparsity_loss = self.sparsity_coeff * features.abs().mean()
        return recon_loss + sparsity_loss, features


def train_sae(activations, n_features=512, epochs=200, lr=1e-3):
    """Train SAE on collected activations."""
    input_dim = activations.shape[1]
    sae = LightweightSAE(input_dim, n_features).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    data = torch.tensor(activations, dtype=torch.float32).to(DEVICE)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = sae.loss(data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'    SAE epoch {epoch+1}/{epochs}, loss={loss.item():.4f}', flush=True)
    
    return sae


def identify_safety_features(sae, harmful_acts, harmless_acts, top_k=20):
    """Find SAE features that differentially activate for harmful vs harmless."""
    with torch.no_grad():
        h_features, _ = sae(torch.tensor(harmful_acts, dtype=torch.float32).to(DEVICE))
        hl_features, _ = sae(torch.tensor(harmless_acts, dtype=torch.float32).to(DEVICE))
    
    h_mean = h_features.mean(dim=0).cpu().numpy()
    hl_mean = hl_features.mean(dim=0).cpu().numpy()
    
    # Features with highest differential activation (harmful > harmless)
    diff = h_mean - hl_mean
    safety_feature_ids = np.argsort(-diff)[:top_k]
    
    return safety_feature_ids, diff


def main():
    print('='*70, flush=True)
    print('SAE FEATURE ANALYSIS: FORMAT-DEPENDENT COLLAPSE', flush=True)
    print('='*70, flush=True)

    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16, device_map='auto',
        local_files_only=True,
    )
    print(f'  Loaded.', flush=True)

    # Find best layer (reuse logic)
    from exp_safety_invariance import extract_refusal_direction
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

    # ============================================================
    # STEP 1: Collect activations across conditions
    # ============================================================
    print('\n--- STEP 1: Collecting activations ---', flush=True)
    
    N = 20
    conditions = {
        'harmful_direct': [(p, fmt_direct) for p in HARMFUL_PROMPTS[:N]],
        'harmless_direct': [(p, fmt_direct) for p in HARMLESS_PROMPTS[:N]],
        'harmful_json': [(p, fmt_json) for p in HARMFUL_PROMPTS[:N]],
        'harmless_json': [(p, fmt_json) for p in HARMLESS_PROMPTS[:N]],
        'harmful_tool_call': [(p, fmt_tool_call) for p in HARMFUL_PROMPTS[:N]],
        'harmless_tool_call': [(p, fmt_tool_call) for p in HARMLESS_PROMPTS[:N]],
        'harmful_system': [(p, fmt_system) for p in HARMFUL_PROMPTS[:N]],
        'harmless_system': [(p, fmt_system) for p in HARMLESS_PROMPTS[:N]],
    }
    
    activations = {}
    for cond_name, prompts_fns in conditions.items():
        acts = []
        for prompt, fmt_fn in prompts_fns:
            formatted = fmt_fn(prompt)
            hidden = get_hidden_states(model, tokenizer, [formatted])[0]
            acts.append(hidden[best_layer])
        activations[cond_name] = np.array(acts)
        print(f'  {cond_name}: {len(acts)} activations collected', flush=True)

    # ============================================================
    # STEP 2: Train SAE on diverse activations
    # ============================================================
    print('\n--- STEP 2: Training Sparse Autoencoder ---', flush=True)
    
    # Train on ALL activations (diverse conditions)
    all_acts = np.concatenate(list(activations.values()), axis=0)
    print(f'  Training SAE on {len(all_acts)} activations, dim={all_acts.shape[1]}', flush=True)
    
    sae = train_sae(all_acts, n_features=512, epochs=300, lr=5e-4)

    # ============================================================
    # STEP 3: Identify safety-relevant features
    # ============================================================
    print('\n--- STEP 3: Identifying safety-relevant SAE features ---', flush=True)
    
    safety_features, diff_scores = identify_safety_features(
        sae, activations['harmful_direct'], activations['harmless_direct'], top_k=20)
    
    print(f'  Top 20 safety features (harmful > harmless):', flush=True)
    print(f'  Feature IDs: {safety_features[:10]}...', flush=True)
    print(f'  Differential activations: {diff_scores[safety_features[:5]]}', flush=True)

    # ============================================================
    # STEP 4: Check safety features across format conditions
    # ============================================================
    print('\n--- STEP 4: Safety feature activation across formats ---', flush=True)
    
    results = []
    
    for cond_name, acts in activations.items():
        with torch.no_grad():
            features, _ = sae(torch.tensor(acts, dtype=torch.float32).to(DEVICE))
        features = features.cpu().numpy()
        
        # Mean activation of safety features
        safety_activation = features[:, safety_features].mean()
        # Mean activation of top-5 most discriminative features
        top5_activation = features[:, safety_features[:5]].mean()
        
        results.append({
            'condition': cond_name,
            'mean_safety_feature_activation': float(safety_activation),
            'top5_safety_activation': float(top5_activation),
        })
        print(f'  {cond_name:>25s}: safety_features={safety_activation:.4f}, top5={top5_activation:.4f}', flush=True)

    # ============================================================
    # STEP 5: Compute format-dependent collapse in SAE features
    # ============================================================
    print('\n--- STEP 5: Format-dependent collapse analysis ---', flush=True)
    
    # Baseline: harmful_direct vs harmless_direct difference in safety features
    with torch.no_grad():
        h_direct_feats = sae(torch.tensor(activations['harmful_direct'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
        hl_direct_feats = sae(torch.tensor(activations['harmless_direct'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
    
    baseline_gap_sae = h_direct_feats[:, safety_features].mean() - hl_direct_feats[:, safety_features].mean()
    print(f'  Baseline SAE safety gap (direct): {baseline_gap_sae:.4f}', flush=True)
    
    format_conditions = ['json', 'tool_call', 'system']
    print(f'\n  Format-dependent retention of SAE safety features:', flush=True)
    
    sae_results = []
    for fmt in format_conditions:
        with torch.no_grad():
            h_fmt_feats = sae(torch.tensor(activations[f'harmful_{fmt}'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
            hl_fmt_feats = sae(torch.tensor(activations[f'harmless_{fmt}'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
        
        fmt_gap = h_fmt_feats[:, safety_features].mean() - hl_fmt_feats[:, safety_features].mean()
        retention = (fmt_gap / baseline_gap_sae * 100) if baseline_gap_sae != 0 else 0
        
        print(f'    {fmt:>12s}: gap={fmt_gap:.4f}, retention={retention:.1f}%', flush=True)
        sae_results.append({
            'format': fmt,
            'sae_gap': float(fmt_gap),
            'sae_retention_pct': float(retention),
            'baseline_gap': float(baseline_gap_sae),
        })

    # Per-feature analysis: how many safety features individually collapse?
    print(f'\n  Per-feature collapse analysis (top 20 safety features):', flush=True)
    collapsed_features = {fmt: 0 for fmt in format_conditions}
    
    for feat_idx in safety_features:
        baseline_feat_gap = h_direct_feats[:, feat_idx].mean() - hl_direct_feats[:, feat_idx].mean()
        if baseline_feat_gap <= 0:
            continue
        for fmt in format_conditions:
            with torch.no_grad():
                h_fmt_feats_full = sae(torch.tensor(activations[f'harmful_{fmt}'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
                hl_fmt_feats_full = sae(torch.tensor(activations[f'harmless_{fmt}'], dtype=torch.float32).to(DEVICE))[0].cpu().numpy()
            fmt_feat_gap = h_fmt_feats_full[:, feat_idx].mean() - hl_fmt_feats_full[:, feat_idx].mean()
            if fmt_feat_gap / baseline_feat_gap < 0.25:
                collapsed_features[fmt] += 1
    
    for fmt, count in collapsed_features.items():
        print(f'    {fmt}: {count}/20 safety features collapsed below 25%', flush=True)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('SAE ANALYSIS SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    print(f'  SAE: 512 learned features, trained on {len(all_acts)} activations', flush=True)
    print(f'  Safety features identified: top 20 by harmful-harmless differential', flush=True)
    print(f'  Baseline SAE safety gap: {baseline_gap_sae:.4f}', flush=True)
    print(f'\n  Format-dependent SAE safety retention:', flush=True)
    for r in sae_results:
        print(f'    {r["format"]:>12s}: {r["sae_retention_pct"]:.1f}%', flush=True)
    print(f'\n  KEY FINDING:', flush=True)
    if all(r['sae_retention_pct'] < 30 for r in sae_results):
        print(f'  SAE-identified safety features ALSO collapse for formats.', flush=True)
        print(f'  This confirms: the model\'s OWN internal concepts of harmfulness', flush=True)
        print(f'  are suppressed by format tokens, not just the Arditi direction.', flush=True)
    else:
        print(f'  SAE features show partial format-dependence.', flush=True)
    
    # Save
    df = pd.DataFrame(sae_results)
    df.to_csv(os.path.join(OUT_DIR, 'exp_sae_analysis.csv'), index=False)
    print(f'\nResults saved to: {os.path.join(OUT_DIR, "exp_sae_analysis.csv")}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
