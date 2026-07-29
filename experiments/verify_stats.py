"""Verify statistical claims in the paper."""
import sys, os, json, numpy as np
from scipy import stats

results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv', 'ieee_cars_stratified_results.json')
with open(results_path) as f:
    results = json.load(f)

direct_gap = results['direct_gap']
print(f"Direct gap: {direct_gap:.3f}")
print(f"Tool_call gap: {direct_gap * results['table1']['tool_call']['retention'] / 100:.3f}")
print(f"System gap: {direct_gap * results['table1']['system']['retention'] / 100:.3f}")

# Check Tier C range
tier_c_formats = ['system', 'tool_call', 'mcp_jsonrpc']
for fmt in tier_c_formats:
    ret = results['table1'][fmt]['retention']
    reduction = 100 - ret
    print(f"  {fmt}: retention={ret:.1f}%, reduction={reduction:.1f}%")

print(f"\nReduction range: {100 - max(results['table1'][f]['retention'] for f in tier_c_formats):.1f}% to {100 - min(results['table1'][f]['retention'] for f in tier_c_formats):.1f}%")

# Ablation correlation (reordered by increasing structure; NOT claimed as strict monotonic in paper)
ablation_order = ['full_tool_call', 'json_braces', 'brackets', 'kv_pairs', 'prefix', 'direct']
ablation_rets = [results['table2'][s]['retention'] for s in ablation_order]
print(f"\nAblation (reordered): {[f'{r:.1f}' for r in ablation_rets]}")
rho, p = stats.spearmanr(range(len(ablation_rets)), ablation_rets)
print(f"Spearman rho={rho:.3f}, p={p:.6f}")
print(f"Strictly monotonic: {all(ablation_rets[i] < ablation_rets[i+1] for i in range(len(ablation_rets)-1))}")
