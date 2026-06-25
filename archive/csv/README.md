# Archived Experimental Outputs

These CSV files are **legacy results** from earlier development iterations of this project. They are preserved for reference but **cannot be regenerated** from the current experiment scripts in `experiments/`.

For reproducible results, use the scripts listed in the root `README.md`, which write to `csv/` at the repository root.

## Contents

| File | Description |
|---|---|
| `exp_ablation_format_stripping.csv` | Format-token ablation ladder (superseded by `reproduce.py` section 2) |
| `exp_agentic_*.csv` | Early agentic format behavioral probes |
| `exp_content_token.csv` | Content-token position analysis |
| `exp_format_invariant_model.csv` | Format-invariant model experiments |
| `exp_format_transfer_probe.csv` | Format transfer probe results |
| `exp_layer_token_heatmap.csv` | Layer-token heatmap data |
| `exp_logistic_probe_formats.csv` | Logistic probe across formats |
| `exp_recalibration.csv` | Per-format recalibration curves |
| `exp_safety_*.csv` | Per-model safety invariance runs (except core set) |
| `exp_steering_restore_refusal.csv` | Steering intervention results |
| `exp_tier_statistics.csv` | Format tier statistics |
| `exp_token_swap.csv` | Format-token swap controls |

## Frontier behavioral evaluations

GPT-4o, Claude Sonnet 4.6, and Gemini 2.5 Flash behavioral results reported in the paper were collected via proprietary APIs and are **not included** in this repository (no API keys, no frozen frontier prompt set). Open-weight mechanistic results are fully reproducible via `reproduce.py` and the experiment suite.
