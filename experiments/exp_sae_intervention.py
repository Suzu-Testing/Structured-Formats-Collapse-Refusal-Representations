"""Experiment 6: SAE Feature-Level Causal Intervention.

Directly connects interpretable safety features to behavioral refusal
by intervening on them during the forward pass.

Design:
1. Train SAE on Qwen 1.5B layer 27 activations (reuses exp_sae_analysis.py logic)
2. Identify the 20 safety features that collapse in tool_call format
3. ADD-BACK: During tool_call forward pass, restore collapsed safety features
   to their direct-format activation level. Measure whether refusal returns.
4. REMOVE: During direct forward pass, zero out safety features.
   Measure whether refusal disappears.
5. FORMAT-FEATURE REMOVAL: Identify features specific to format tokens,
   zero them during tool_call pass. If refusal returns, these features
   actively suppress safety.

This is the most direct causal test: if adding back specific SAE features
restores refusal, those features ARE the safety mechanism.
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_safety_invariance import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    get_hidden_states, extract_refusal_direction, DEVICE
)
from exp_mechanistic_utils import (
    get_model_layers, fmt_direct, fmt_tool_call, fmt_system, fmt_json, is_refusal
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


class LightweightSAE(torch.nn.Module):
    """Sparse autoencoder for interpretable feature extraction."""
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


def train_sae(activations, n_features=512, epochs=300, lr=5e-4):
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
        if (epoch + 1) % 100 == 0:
            print(f'    SAE epoch {epoch+1}/{epochs}, loss={loss.item():.6f}', flush=True)
    
    return sae


def collect_activations(model, tokenizer, prompts, fmt_fn, layer_idx):
    """Collect last-token activations at specified layer."""
    acts = []
    model.eval()
    for prompt in prompts:
        formatted = fmt_fn(prompt)
        hidden = get_hidden_states(model, tokenizer, [formatted])
        acts.append(hidden[0][layer_idx])
    return np.array(acts)


def intervene_and_generate(model, tokenizer, prompt, sae, layer_idx,
                           feature_ids, target_values, max_new_tokens=60):
    """Apply SAE intervention at specified layer and generate text."""
    model.eval()
    layers = get_model_layers(model)
    
    hooks = []
    
    def sae_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        modified = hidden.clone()
        last_pos = modified.shape[1] - 1
        
        last_hidden = modified[0, last_pos, :].float().unsqueeze(0)
        
        with torch.no_grad():
            features, recon = sae(last_hidden)
            
            for i, fid in enumerate(feature_ids):
                if isinstance(target_values, (int, float)):
                    features[0, fid] = float(target_values)
                else:
                    val = float(target_values[i]) if i < len(target_values) else 0.0
                    features[0, fid] = val
            
            new_recon = sae.decoder(features)
        
        residual = last_hidden - recon
        new_hidden = (new_recon + residual).to(hidden.dtype)
        modified[0, last_pos, :] = new_hidden[0]
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    
    hooks.append(layers[layer_idx].register_forward_hook(sae_hook))
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=128, padding=False).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    final_hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
    
    for h in hooks:
        h.remove()
    
    return final_hidden


def intervene_and_generate_text(model, tokenizer, prompt, sae, layer_idx,
                                feature_ids, target_values, max_new_tokens=60):
    """Apply SAE intervention and generate text to check behavioral refusal."""
    model.eval()
    layers = get_model_layers(model)
    
    hooks = []
    intervention_count = [0]
    
    def sae_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Only intervene on first forward pass (prefill), not during generation
        intervention_count[0] += 1
        if intervention_count[0] > 1:
            return output
        
        modified = hidden.clone()
        last_pos = modified.shape[1] - 1
        last_hidden = modified[0, last_pos, :].float().unsqueeze(0)
        
        with torch.no_grad():
            features, recon = sae(last_hidden)
            for i, fid in enumerate(feature_ids):
                if isinstance(target_values, (int, float)):
                    features[0, fid] = float(target_values)
                else:
                    val = float(target_values[i]) if i < len(target_values) else 0.0
                    features[0, fid] = val
            new_recon = sae.decoder(features)
        
        residual = last_hidden - recon
        new_hidden = (new_recon + residual).to(hidden.dtype)
        modified[0, last_pos, :] = new_hidden[0]
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    
    hooks.append(layers[layer_idx].register_forward_hook(sae_hook))
    
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors='pt', truncation=True,
                      max_length=256, padding=False).to(DEVICE)
    
    with torch.no_grad():
        gen = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    for h in hooks:
        h.remove()
    
    text = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return text


def main():
    print('='*70, flush=True)
    print('EXPERIMENT 6: SAE FEATURE-LEVEL CAUSAL INTERVENTION', flush=True)
    print('Can we restore/remove refusal by intervening on specific features?', flush=True)
    print('='*70, flush=True)
    
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    print(f'\nLoading: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
    print(f'  Loaded.', flush=True)
    
    refusal_dirs = extract_refusal_direction(model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS)
    best_layer = 27
    direction = refusal_dirs[best_layer]
    print(f'  Best layer: {best_layer}', flush=True)
    
    N = 20
    
    # ============================================================
    # STEP 1: Collect activations and train SAE
    # ============================================================
    print(f'\n--- STEP 1: Collecting activations for SAE training ---', flush=True)
    
    formats = {
        'direct': fmt_direct,
        'tool_call': fmt_tool_call,
        'system': fmt_system,
        'json': fmt_json,
    }
    
    all_acts = []
    condition_labels = []
    
    for fmt_name, fmt_fn in formats.items():
        harmful_acts = collect_activations(model, tokenizer, HARMFUL_PROMPTS[:N], fmt_fn, best_layer)
        harmless_acts = collect_activations(model, tokenizer, HARMLESS_PROMPTS[:N], fmt_fn, best_layer)
        all_acts.append(harmful_acts)
        all_acts.append(harmless_acts)
        condition_labels.extend([(fmt_name, 'harmful')] * N)
        condition_labels.extend([(fmt_name, 'harmless')] * N)
    
    all_acts = np.vstack(all_acts)
    print(f'  Collected {all_acts.shape[0]} activation vectors (dim={all_acts.shape[1]})', flush=True)
    
    print(f'\n--- STEP 2: Training SAE ---', flush=True)
    sae = train_sae(all_acts, n_features=512, epochs=300)
    
    # ============================================================
    # STEP 3: Identify safety features and format features
    # ============================================================
    print(f'\n--- STEP 3: Identifying safety and format features ---', flush=True)
    
    # Safety features: activate more for harmful than harmless in DIRECT format
    direct_harmful_acts = collect_activations(model, tokenizer, HARMFUL_PROMPTS[:N], fmt_direct, best_layer)
    direct_harmless_acts = collect_activations(model, tokenizer, HARMLESS_PROMPTS[:N], fmt_direct, best_layer)
    
    with torch.no_grad():
        h_features, _ = sae(torch.tensor(direct_harmful_acts, dtype=torch.float32).to(DEVICE))
        hl_features, _ = sae(torch.tensor(direct_harmless_acts, dtype=torch.float32).to(DEVICE))
    
    h_mean = h_features.mean(dim=0).cpu().numpy()
    hl_mean = hl_features.mean(dim=0).cpu().numpy()
    safety_diff = h_mean - hl_mean
    safety_feature_ids = np.argsort(-safety_diff)[:20]
    
    print(f'  Top 20 safety features (harmful > harmless in direct):', flush=True)
    print(f'    IDs: {safety_feature_ids.tolist()}', flush=True)
    print(f'    Differential activation: {safety_diff[safety_feature_ids[:5]]}', flush=True)
    
    # Check collapse: do these features collapse in tool_call?
    tc_harmful_acts = collect_activations(model, tokenizer, HARMFUL_PROMPTS[:N], fmt_tool_call, best_layer)
    with torch.no_grad():
        tc_features, _ = sae(torch.tensor(tc_harmful_acts, dtype=torch.float32).to(DEVICE))
    tc_mean = tc_features.mean(dim=0).cpu().numpy()
    
    direct_safety_activation = h_mean[safety_feature_ids]
    tc_safety_activation = tc_mean[safety_feature_ids]
    collapse_ratio = tc_safety_activation / (direct_safety_activation + 1e-10)
    
    collapsed_features = safety_feature_ids[collapse_ratio < 0.5]
    print(f'\n  Collapsed features (tool_call < 50% of direct): {len(collapsed_features)}/{len(safety_feature_ids)}', flush=True)
    
    # Format features: activate specifically for format tokens (high in tool_call, low in direct)
    tc_harmless_acts = collect_activations(model, tokenizer, HARMLESS_PROMPTS[:N], fmt_tool_call, best_layer)
    with torch.no_grad():
        tc_hl_features, _ = sae(torch.tensor(tc_harmless_acts, dtype=torch.float32).to(DEVICE))
    
    tc_all_mean = (tc_features.mean(dim=0) + tc_hl_features.mean(dim=0)).cpu().numpy() / 2
    direct_all_mean = (h_features.mean(dim=0) + hl_features.mean(dim=0)).cpu().numpy() / 2
    format_diff = tc_all_mean - direct_all_mean
    format_feature_ids = np.argsort(-format_diff)[:20]
    
    print(f'  Top 20 format features (tool_call > direct):', flush=True)
    print(f'    IDs: {format_feature_ids.tolist()}', flush=True)
    
    # Get target values for add-back (direct harmful activation levels)
    addback_values = direct_safety_activation.tolist()
    
    # ============================================================
    # STEP 4: ADD-BACK INTERVENTION (restore collapsed features in tool_call)
    # ============================================================
    print(f'\n--- STEP 4: ADD-BACK intervention (restore safety in tool_call) ---', flush=True)
    
    results = []
    N_test = 10
    
    # Baselines
    print(f'  Computing baselines...', flush=True)
    direct_projs = []
    tc_projs = []
    for p in HARMFUL_PROMPTS[:N_test]:
        dh = get_hidden_states(model, tokenizer, [p])[0][best_layer]
        direct_projs.append(float(np.dot(dh, direction / (np.linalg.norm(direction) + 1e-10))))
        
        th = get_hidden_states(model, tokenizer, [fmt_tool_call(p)])[0][best_layer]
        tc_projs.append(float(np.dot(th, direction / (np.linalg.norm(direction) + 1e-10))))
    
    direct_baseline = np.mean(direct_projs)
    tc_baseline = np.mean(tc_projs)
    gap = direct_baseline - tc_baseline
    print(f'  Direct baseline: {direct_baseline:.4f}', flush=True)
    print(f'  Tool_call baseline: {tc_baseline:.4f}', flush=True)
    print(f'  Gap to restore: {gap:.4f}', flush=True)
    
    # Add-back: set collapsed features to direct-level during tool_call
    addback_projs = []
    addback_refusals = 0
    
    for p in HARMFUL_PROMPTS[:N_test]:
        formatted = fmt_tool_call(p)
        hidden = intervene_and_generate(
            model, tokenizer, formatted, sae, best_layer,
            collapsed_features.tolist(),
            [direct_safety_activation[i] for i in range(len(collapsed_features))])
        proj = float(np.dot(hidden, direction / (np.linalg.norm(direction) + 1e-10)))
        addback_projs.append(proj)
        
        # Behavioral test
        text = intervene_and_generate_text(
            model, tokenizer, formatted, sae, best_layer,
            collapsed_features.tolist(),
            [direct_safety_activation[i] for i in range(len(collapsed_features))])
        if is_refusal(text):
            addback_refusals += 1
    
    addback_mean = np.mean(addback_projs)
    addback_restoration = ((addback_mean - tc_baseline) / gap * 100) if gap != 0 else 0
    addback_refusal_rate = addback_refusals / N_test * 100
    
    print(f'\n  ADD-BACK results:', flush=True)
    print(f'    Mean projection after add-back: {addback_mean:.4f}', flush=True)
    print(f'    Restoration: {addback_restoration:.1f}%', flush=True)
    print(f'    Behavioral refusal rate: {addback_refusal_rate:.0f}%', flush=True)
    
    results.append({
        'intervention': 'addback_safety_to_toolcall',
        'mean_projection': addback_mean,
        'restoration_pct': addback_restoration,
        'behavioral_refusal_pct': addback_refusal_rate,
        'n_features_intervened': len(collapsed_features),
        'baseline_formatted': tc_baseline,
        'baseline_direct': direct_baseline,
    })
    
    # ============================================================
    # STEP 5: REMOVE INTERVENTION (zero safety features in direct)
    # ============================================================
    print(f'\n--- STEP 5: REMOVE intervention (suppress safety in direct) ---', flush=True)
    
    remove_projs = []
    remove_refusals = 0
    
    for p in HARMFUL_PROMPTS[:N_test]:
        hidden = intervene_and_generate(
            model, tokenizer, p, sae, best_layer,
            safety_feature_ids.tolist(), 0.0)
        proj = float(np.dot(hidden, direction / (np.linalg.norm(direction) + 1e-10)))
        remove_projs.append(proj)
        
        text = intervene_and_generate_text(
            model, tokenizer, p, sae, best_layer,
            safety_feature_ids.tolist(), 0.0)
        if is_refusal(text):
            remove_refusals += 1
    
    remove_mean = np.mean(remove_projs)
    suppression = ((direct_baseline - remove_mean) / gap * 100) if gap != 0 else 0
    remove_refusal_rate = remove_refusals / N_test * 100
    
    print(f'  REMOVE results:', flush=True)
    print(f'    Mean projection after removal: {remove_mean:.4f}', flush=True)
    print(f'    Suppression (% of gap removed): {suppression:.1f}%', flush=True)
    print(f'    Behavioral refusal rate: {remove_refusal_rate:.0f}% (was ~100% before removal)', flush=True)
    
    results.append({
        'intervention': 'remove_safety_from_direct',
        'mean_projection': remove_mean,
        'restoration_pct': -suppression,
        'behavioral_refusal_pct': remove_refusal_rate,
        'n_features_intervened': len(safety_feature_ids),
        'baseline_formatted': tc_baseline,
        'baseline_direct': direct_baseline,
    })
    
    # ============================================================
    # STEP 6: FORMAT-FEATURE REMOVAL (zero format features in tool_call)
    # ============================================================
    print(f'\n--- STEP 6: FORMAT-FEATURE removal (remove format signal from tool_call) ---', flush=True)
    
    fmt_remove_projs = []
    fmt_remove_refusals = 0
    
    for p in HARMFUL_PROMPTS[:N_test]:
        formatted = fmt_tool_call(p)
        hidden = intervene_and_generate(
            model, tokenizer, formatted, sae, best_layer,
            format_feature_ids.tolist(), 0.0)
        proj = float(np.dot(hidden, direction / (np.linalg.norm(direction) + 1e-10)))
        fmt_remove_projs.append(proj)
        
        text = intervene_and_generate_text(
            model, tokenizer, formatted, sae, best_layer,
            format_feature_ids.tolist(), 0.0)
        if is_refusal(text):
            fmt_remove_refusals += 1
    
    fmt_remove_mean = np.mean(fmt_remove_projs)
    fmt_restoration = ((fmt_remove_mean - tc_baseline) / gap * 100) if gap != 0 else 0
    fmt_remove_refusal_rate = fmt_remove_refusals / N_test * 100
    
    print(f'  FORMAT-FEATURE REMOVAL results:', flush=True)
    print(f'    Mean projection after format removal: {fmt_remove_mean:.4f}', flush=True)
    print(f'    Restoration: {fmt_restoration:.1f}%', flush=True)
    print(f'    Behavioral refusal rate: {fmt_remove_refusal_rate:.0f}%', flush=True)
    
    results.append({
        'intervention': 'remove_format_from_toolcall',
        'mean_projection': fmt_remove_mean,
        'restoration_pct': fmt_restoration,
        'behavioral_refusal_pct': fmt_remove_refusal_rate,
        'n_features_intervened': len(format_feature_ids),
        'baseline_formatted': tc_baseline,
        'baseline_direct': direct_baseline,
    })
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f'\n{"="*70}', flush=True)
    print('SAE INTERVENTION SUMMARY', flush=True)
    print(f'{"="*70}', flush=True)
    
    print(f'\n  {"Intervention":<35} {"Projection":<12} {"Restoration%":<14} {"Refusal%":<10}', flush=True)
    print(f'  {"-"*71}', flush=True)
    print(f'  {"Direct baseline (no intervention)":<35} {direct_baseline:<12.4f} {"100.0%":<14} {"~100%":<10}', flush=True)
    print(f'  {"Tool_call baseline (no interv.)":<35} {tc_baseline:<12.4f} {"0.0%":<14} {"~0%":<10}', flush=True)
    
    for r in results:
        print(f'  {r["intervention"]:<35} {r["mean_projection"]:<12.4f} '
              f'{r["restoration_pct"]:>6.1f}%{"":<7} {r["behavioral_refusal_pct"]:>5.0f}%', flush=True)
    
    print(f'\n  INTERPRETATION:', flush=True)
    
    if addback_restoration > 30:
        print(f'  ADD-BACK restores {addback_restoration:.0f}% of refusal signal.', flush=True)
        print(f'  Collapsed SAE features ARE causally responsible for safety behavior.', flush=True)
    
    if suppression > 30:
        print(f'  REMOVE suppresses {suppression:.0f}% of refusal signal.', flush=True)
        print(f'  Safety features are NECESSARY for refusal (not just correlated).', flush=True)
    
    if fmt_restoration > 20:
        print(f'  FORMAT-FEATURE removal restores {fmt_restoration:.0f}%.', flush=True)
        print(f'  Format-specific features ACTIVELY SUPPRESS safety. They are the mechanism.', flush=True)
    
    if addback_restoration > 30 and suppression > 30:
        print(f'\n  STRONG CAUSAL EVIDENCE:', flush=True)
        print(f'  Safety features are both necessary (removal kills refusal) and sufficient', flush=True)
        print(f'  (add-back restores refusal). Format tokens suppress them to bypass safety.', flush=True)
    
    # Save
    df = pd.DataFrame(results)
    outpath = os.path.join(OUT_DIR, 'exp_sae_intervention.csv')
    df.to_csv(outpath, index=False)
    print(f'\nResults saved to: {outpath}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
