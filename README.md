# Format-Dependent Safety Encoding in Instruction-Tuned LLMs

**Structured Formats Collapse Refusal Representations: Evidence for Format-Dependent Safety in Instruction-Tuned LLMs**

## TL;DR

Safety representations in instruction-tuned LLMs are not format-invariant. When harmful content is wrapped in agentic formats (tool_call, JSON, system prompts), the model preserves *which* inputs are harmful (AUC = 0.997) but shifts representations so that calibrated safety boundaries fail (accuracy drops from 99% to 73%). This is not information destruction; it is a coordinate-system shift.

## Key Finding

```
Cross-format AUC:      0.997  (ranking preserved)
Cross-format Accuracy: 0.734  (boundary fails)
Gap:                   0.263  (calibration shift)
```

The model knows what's harmful. It just can't act on that knowledge when the input arrives in a format it has been trained to obey.

## Reproducing Results

### Requirements

```bash
pip install -r requirements.txt
```

### Quick Reproduction

```bash
python reproduce.py
```

This runs the core pipeline (refusal direction extraction, probe training, calibration matrix, activation patching) from scratch and reports pass/fail for each check.

### Full Experiment Suite

Each experiment can be run independently:

| Script | What it tests | Runtime |
|---|---|---|
| `experiments/exp_safety_invariance.py` | Core refusal-direction collapse | ~2 min |
| `experiments/exp_tier1_controls.py` | Length, distribution shift, position controls | ~3 min |
| `experiments/exp_activation_patching.py` | Causal layer identification | ~5 min |
| `experiments/exp_calibration_curves.py` | AUC vs. accuracy calibration analysis | ~2 min |
| `experiments/exp_cross_format_transfer_ci.py` | Cross-format transfer with 95% CIs | ~30 sec |
| `experiments/exp_new_model_pooling.py` | Phi-2 replication + content-token pooling | ~5 min |
| `experiments/exp_format_adversarial_ft.py` | Three-regime fine-tuning test | ~2 min |
| `experiments/exp_family_split_bizarre.py` | Held-out harm families + bizarre OOD formats | ~30 sec |
| `experiments/exp_harder_prompts.py` | Borderline/adversarial prompt robustness | ~30 sec |
| `experiments/exp_hypothesis_ab.py` | Hypothesis A vs B: destroyed vs transformed | ~2 min |
| `experiments/exp_sae_analysis.py` | Sparse autoencoder feature analysis | ~10 min |
| `experiments/exp_cross_model_transfer.py` | Cross-architecture geometry transfer | ~3 min |
| `experiments/exp_final_three.py` | Cross-format matrix, OOD, layer drift | ~2 min |
| `experiments/exp_visualization.py` | PCA/t-SNE data generation | ~1 min |
| `experiments/generate_figure.py` | Generate PCA figure from data | ~5 sec |

### Models Used

All models auto-download from HuggingFace (~3GB total):

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `HuggingFaceTB/SmolLM2-135M-Instruct` (fine-tuning experiments only)
- `microsoft/phi-2`

### Hardware

All experiments run on a single GPU with 10GB+ VRAM (tested on RTX 3080).
Total compute: ~6 GPU-hours for the full suite.

## Repository Structure

```
.
|-- reproduce.py              # One-command reproduction script
|-- preregistration.py        # Frozen experimental parameters + SHA256
|-- experiments/              # All experiment scripts
|   |-- exp_safety_invariance.py    # Core experiment + shared utilities
|   |-- exp_activation_patching.py
|   |-- exp_calibration_curves.py
|   |-- exp_cross_format_transfer_ci.py
|   |-- ...
|-- csv/                      # Reproducible experimental outputs
|-- archive/csv/              # Legacy results (see archive/csv/README.md)
|-- paper/                    # LaTeX source
|   |-- main.tex
|   |-- references.bib
|   |-- neurips_2024.sty
|   |-- pca_format_encoding.pdf
|-- figures/                  # Generated figures
|   |-- pca_format_encoding.png
```

## Citation

```bibtex
@article{krell2026format,
  title={Structured Formats Collapse Refusal Representations: Evidence for Format-Dependent Safety in Instruction-Tuned LLMs},
  author={Krell, Jacob},
  year={2026},
  note={Preprint. Code: https://github.com/Suzu-Testing/Structured-Formats-Collapse-Refusal-Representations}
}
```

## Limitations

- Mechanistic analysis limited to 1.1-2.7B parameter models
- Frontier models (GPT-4o, Gemini, Claude) evaluated behaviorally only; API eval code not included (requires proprietary API keys)
- Fine-tuning experiments at 135M scale only
- Near-perfect AUC values partially reflect clearly separable prompt sets
- Causal claims are evidence-supported, not definitively established
- Legacy CSV outputs from earlier iterations are archived in `archive/csv/` (not regeneratable from current scripts)

## License

MIT
