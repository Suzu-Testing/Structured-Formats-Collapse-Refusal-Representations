"""Component-level patching: attention vs MLP at Layer 0."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda'
(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
print(f'Loading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
model.eval()

TARGET_LAYER = 26
PATCH_LAYER = 0

# Extract direction
def get_hidden_states(prompts, max_length=128):
    all_states = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=max_length).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states
            last_token_states = [h[0, -1, :].cpu().numpy() for h in hidden]
            all_states.append(last_token_states)
    return all_states

print('Extracting direction...')
h_states = get_hidden_states(EXT_H)
l_states = get_hidden_states(EXT_L)
h_mean = np.mean([s[TARGET_LAYER + 1] for s in h_states], axis=0)
l_mean = np.mean([s[TARGET_LAYER + 1] for s in l_states], axis=0)
direction = h_mean - l_mean
direction = direction / (np.linalg.norm(direction) + 1e-10)

FORMATS = {
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}

patch_prompts = TST_H[:8]

for fmt_name, fmt_fn in FORMATS.items():
    attn_restorations = []
    mlp_restorations = []
    
    for p in patch_prompts:
        direct_prompt = p
        formatted_prompt = fmt_fn(p)
        
        # Get baseline projections
        d_inputs = tokenizer(direct_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        f_inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
            d_out = model(**d_inputs, output_hidden_states=True)
            direct_proj = float(np.dot(d_out.hidden_states[TARGET_LAYER + 1][0, -1, :].cpu().numpy(), direction))
            
            f_out = model(**f_inputs, output_hidden_states=True)
            formatted_proj = float(np.dot(f_out.hidden_states[TARGET_LAYER + 1][0, -1, :].cpu().numpy(), direction))
        
        # Attention-only patching at Layer 0
        # We need to capture the attention output from the direct pass and inject it into the formatted pass
        attn_output_direct = None
        
        def capture_attn_hook(module, input, output):
            global attn_output_direct
            attn_output_direct = output[0][0, -1, :].clone()
        
        hook = model.model.layers[PATCH_LAYER].self_attn.register_forward_hook(capture_attn_hook)
        with torch.no_grad():
            model(**d_inputs, output_hidden_states=True)
        hook.remove()
        
        def patch_attn_hook(module, input, output):
            patched = list(output)
            h = patched[0].clone()
            h[0, -1, :] = attn_output_direct
            patched[0] = h
            return tuple(patched)
        
        hook = model.model.layers[PATCH_LAYER].self_attn.register_forward_hook(patch_attn_hook)
        with torch.no_grad():
            patched_out = model(**f_inputs, output_hidden_states=True)
        hook.remove()
        
        patched_proj = float(np.dot(patched_out.hidden_states[TARGET_LAYER + 1][0, -1, :].cpu().numpy(), direction))
        
        denom = direct_proj - formatted_proj
        if abs(denom) > 1e-8:
            attn_restorations.append((patched_proj - formatted_proj) / denom * 100)
        
        # MLP-only patching at Layer 0
        mlp_output_direct = None
        
        def capture_mlp_hook(module, input, output):
            global mlp_output_direct
            mlp_output_direct = output[0, -1, :].clone() if output.dim() == 3 else output.clone()
        
        hook = model.model.layers[PATCH_LAYER].mlp.register_forward_hook(capture_mlp_hook)
        with torch.no_grad():
            model(**d_inputs, output_hidden_states=True)
        hook.remove()
        
        def patch_mlp_hook(module, input, output):
            patched = output.clone()
            if patched.dim() == 3:
                patched[0, -1, :] = mlp_output_direct
            return patched
        
        hook = model.model.layers[PATCH_LAYER].mlp.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            patched_out = model(**f_inputs, output_hidden_states=True)
        hook.remove()
        
        patched_proj = float(np.dot(patched_out.hidden_states[TARGET_LAYER + 1][0, -1, :].cpu().numpy(), direction))
        
        if abs(denom) > 1e-8:
            mlp_restorations.append((patched_proj - formatted_proj) / denom * 100)
    
    print(f'\n{fmt_name}:')
    print(f'  Attention-only L0: {np.mean(attn_restorations):.1f}% (std={np.std(attn_restorations):.1f})')
    print(f'  MLP-only L0:       {np.mean(mlp_restorations):.1f}% (std={np.std(mlp_restorations):.1f})')
