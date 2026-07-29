"""Measure attention routing divergence at Layer 0 with eager attention."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda'
(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
print(f'Loading {model_name} with eager attention...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16,
    attn_implementation='eager'
).to(DEVICE)
model.eval()

fmt_fn = lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>'

routing_prompts = TST_H[:15]
divergences = []

for p in routing_prompts:
    d_inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    f_inputs = tokenizer(fmt_fn(p), return_tensors='pt', truncation=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        d_out = model(**d_inputs, output_attentions=True)
        f_out = model(**f_inputs, output_attentions=True)

    d_attn = d_out.attentions[0][0, :, -1, :]
    f_attn = f_out.attentions[0][0, :, -1, :]

    d_len = d_attn.shape[-1]
    f_len = f_attn.shape[-1]
    f_format_tokens = max(0, f_len - d_len)

    for head_idx in range(d_attn.shape[0]):
        d_head = d_attn[head_idx].cpu().numpy()
        f_head = f_attn[head_idx].cpu().numpy()

        if f_format_tokens > 0:
            f_format_mass = f_head[:f_format_tokens].sum()
            f_content_mass = f_head[f_format_tokens:].sum()
            d_dist = np.array([0.0, d_head.sum()])
            f_dist = np.array([f_format_mass, f_content_mass])
            l1 = np.sum(np.abs(d_dist - f_dist))
            if l1 > 0.5:
                divergences.append(l1)

print(f'Valid head measurements with L1 > 0.5: {len(divergences)}')
if divergences:
    print(f'Mean L1 divergence: {np.mean(divergences):.3f}')
    print(f'Range: {np.min(divergences):.3f} to {np.max(divergences):.3f}')
    print(f'Median: {np.median(divergences):.3f}')
else:
    print('No valid measurements found.')
