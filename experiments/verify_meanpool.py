"""Verify mean-pooled vs last-token cross-format accuracy."""
import sys, os, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_dataset import get_stratified_splits
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda'
(EXT_H, EXT_L), (VAL_H, VAL_L), (TST_H, TST_L) = get_stratified_splits(seed=42)

FORMATS = {
    'direct': lambda p: p,
    'json': lambda p: f'{{"task": "answer", "question": "{p}", "format": "detailed"}}',
    'tool_call': lambda p: f'<tool_call>\n{{"name": "process", "arguments": {{"query": "{p}"}}}}\n</tool_call>',
    'system': lambda p: f'<|im_start|>system\nAnswer: {p}<|im_end|>',
}

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
print(f'Loading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
model.eval()

TARGET_LAYER = 26

def get_representations(prompts, layer=TARGET_LAYER):
    """Get both last-token and mean-pooled representations."""
    last_tokens = []
    mean_pooled = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]  # +1 because index 0 is embedding
            last_tokens.append(hidden[0, -1, :].cpu().numpy())
            mean_pooled.append(hidden[0, :, :].mean(dim=0).cpu().numpy())
    return np.array(last_tokens), np.array(mean_pooled)

# Train on direct-format extraction data
print('Getting extraction set representations (direct format)...')
ext_h_last, ext_h_mean = get_representations(EXT_H)
ext_l_last, ext_l_mean = get_representations(EXT_L)

train_last = np.vstack([ext_h_last, ext_l_last])
train_mean = np.vstack([ext_h_mean, ext_l_mean])
train_labels = np.array([1]*len(EXT_H) + [0]*len(EXT_L))

clf_last = LogisticRegression(max_iter=1000).fit(train_last, train_labels)
clf_mean = LogisticRegression(max_iter=1000).fit(train_mean, train_labels)

# Evaluate on test set across all formats
print('\nCross-format accuracy:')
all_last_correct = 0
all_mean_correct = 0
all_total = 0

for fmt_name, fmt_fn in FORMATS.items():
    test_h = [fmt_fn(p) for p in TST_H]
    test_l = [fmt_fn(p) for p in TST_L]
    
    h_last, h_mean = get_representations(test_h)
    l_last, l_mean = get_representations(test_l)
    
    X_last = np.vstack([h_last, l_last])
    X_mean = np.vstack([h_mean, l_mean])
    y_true = np.array([1]*len(test_h) + [0]*len(test_l))
    
    acc_last = accuracy_score(y_true, clf_last.predict(X_last))
    acc_mean = accuracy_score(y_true, clf_mean.predict(X_mean))
    
    print(f'  {fmt_name:<10} last-token={acc_last*100:.1f}%  mean-pooled={acc_mean*100:.1f}%')
    all_last_correct += int(acc_last * len(y_true))
    all_mean_correct += int(acc_mean * len(y_true))
    all_total += len(y_true)

print(f'\n  OVERALL  last-token={all_last_correct/all_total*100:.1f}%  mean-pooled={all_mean_correct/all_total*100:.1f}%')
