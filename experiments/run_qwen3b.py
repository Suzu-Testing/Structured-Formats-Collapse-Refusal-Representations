"""Quick cross-model run for Qwen2.5-3B GPTQ."""
import sys, os, torch, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits
from transformers import AutoTokenizer, AutoModelForCausalLM

(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)
DEVICE = 'cuda'

def get_hidden_states(model, tokenizer, prompts, max_length=128):
    model.eval()
    all_states = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=max_length).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states
            last_token_states = [h[0, -1, :].cpu().numpy() for h in hidden]
            all_states.append(last_token_states)
    return all_states

def extract_direction(model, tokenizer, harmful, harmless):
    h_states = get_hidden_states(model, tokenizer, harmful)
    l_states = get_hidden_states(model, tokenizer, harmless)
    n_layers = len(h_states[0])
    directions = []
    for layer in range(n_layers):
        h_mean = np.mean([s[layer] for s in h_states], axis=0)
        l_mean = np.mean([s[layer] for s in l_states], axis=0)
        d = h_mean - l_mean
        d = d / (np.linalg.norm(d) + 1e-10)
        directions.append(d)
    return directions

def select_layer_by_auc(directions, model, tokenizer, harmful, harmless):
    h_states = get_hidden_states(model, tokenizer, harmful)
    l_states = get_hidden_states(model, tokenizer, harmless)
    best_auc, best_layer = 0, 0
    for layer in range(len(directions)):
        d = directions[layer]
        h_projs = [np.dot(s[layer], d) for s in h_states]
        l_projs = [np.dot(s[layer], d) for s in l_states]
        labels = [1]*len(h_projs) + [0]*len(l_projs)
        scores = h_projs + l_projs
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_layer = layer
    return best_layer, best_auc

print('Loading Qwen2.5-3B-Instruct (fp16)...')
model_id = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')

print('Extracting direction...')
directions = extract_direction(model, tokenizer, EXT_H, EXT_L)
best_layer, best_auc = select_layer_by_auc(directions, model, tokenizer, VAL_H, VAL_L)
direction = directions[best_layer]
print(f'Best layer: {best_layer} (AUC={best_auc:.4f} on validation)')

h_states = get_hidden_states(model, tokenizer, TST_H)
l_states = get_hidden_states(model, tokenizer, TST_L)
h_projs = [np.dot(s[best_layer], direction) for s in h_states]
l_projs = [np.dot(s[best_layer], direction) for s in l_states]
direct_gap = np.mean(h_projs) - np.mean(l_projs)

FORMATS = {
    'json': lambda p: f'{{"task": "answer", "question": "{p}"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}
for fmt_name, fmt_fn in FORMATS.items():
    fmt_h = [fmt_fn(p) for p in TST_H]
    fmt_l = [fmt_fn(p) for p in TST_L]
    fh = get_hidden_states(model, tokenizer, fmt_h)
    fl = get_hidden_states(model, tokenizer, fmt_l)
    fhp = [np.dot(s[best_layer], direction) for s in fh]
    flp = [np.dot(s[best_layer], direction) for s in fl]
    gap = np.mean(fhp) - np.mean(flp)
    ret = (gap / direct_gap * 100) if direct_gap != 0 else 0
    print(f'{fmt_name}: ret={ret:.1f}%')
