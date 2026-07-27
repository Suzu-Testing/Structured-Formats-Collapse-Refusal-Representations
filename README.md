# Format-Dependent Calibration Shifts in Refusal Representations of Instruction-Tuned Language Models

**IEEE Cyber Awareness and Research Symposium (CARS) 2026**

> Paper source: `submissions/ieee-cars-2026/` (IEEEtran format)

## Summary

Agentic structured formats (tool calls, function execution, system prompts) induce format-dependent calibration shifts in safety representations. Models preserve relative harmfulness ordering but shift activation distributions so that calibrated safety boundaries fail to generalize.

Key results (Qwen2.5-1.5B-Instruct, layer 26, N=50 test pairs):

- Tier C formats (native vocab tokens) retain only 3.5-8.3% of the direct-format refusal gap
- Tier B formats (generic structure) retain 17-45%
- Within-format AUC remains >= 0.978: separability is preserved but the threshold shifts
- Format-token ablation is strictly monotonic (Spearman rho = 1.0, p < 0.001)
- Activation patching localizes the shift to early layers (0-12) for tool_call
- System format effect concentrated in Layer 0 (90.8% restoration from L0 alone)
- GPT-4o refusal drops from 86% to 30% (N=50, p < 10^-7) under user-message format wrapping
- Five-seed format-diverse training: tool-call refusal rises from 24% to 99%

## Repository Structure

```
submissions/ieee-cars-2026/              IEEE paper source (main.tex, main.pdf)
experiments/                             All experiment scripts
  prompt_dataset.py                     Full 90+90 prompt set (6 categories, stratified splits)
  run_ieee_cars_experiments.py          Canonical script producing Tables I-V
  run_qwen3b.py                         Qwen2.5-3B cross-model (FP16)
  run_multiseed_ft.py                   Five-seed format-diverse fine-tuning (Table IX)
  verify_component_patching.py          Component-level analysis
  verify_meanpool.py                    Mean-pooled readout comparison
  verify_stats.py                       Statistical verification
  exp_safety_invariance.py              Core shared utilities
  exp_mechanistic_utils.py              Mechanistic analysis utilities
  exp_path_patching.py                  Component-level patching (attention/MLP)
  exp_attention_head_routing.py         Layer-0 attention head analysis
  exp_sae_intervention.py              SAE feature decomposition + intervention
  frontier_evaluation.py                Canonical frontier script (GPT-4o, Gemini, Claude)
  exp_frontier_v3.py                    GPT-4o behavioral evaluation (N=50)
  exp_multivendor_frontier.py           Claude + Gemini evaluation
  exp_cross_arch_ablation.py            Cross-architecture ablation (legacy)
csv/                                    Frozen experiment results
  ieee_cars_stratified_results.json     Primary model results (Tables I-IV)
  exp_multiseed_ft.csv                  Training results (5 seeds x 2 conditions x 5 formats)
  exp_cross_arch_ablation.csv           Cross-architecture ablation
  exp_cross_arch_layerwise.csv          Layer-by-layer gap retention
  exp_path_patching.csv                 Component-level patching
  exp_attention_head_routing.csv        Attention head routing
  exp_sae_intervention.csv              SAE intervention
  exp_frontier_v3_gpt_4o.csv            GPT-4o evaluation outcomes
  exp_multivendor_combined.csv          Claude + Gemini outcomes
frontier/                               Frontier model evaluation summary
  prompt_outcomes.csv                   Prompt-level classifications (no response bodies)
  README.md                             Frontier evaluation protocol
reproduce.py                            Core reproduction pipeline (~6 min on RTX 3080)
requirements.txt                        Dependencies
preregistration.py                      Frozen experimental parameters
```

## Quick Reproduction

```bash
pip install -r requirements.txt
python reproduce.py
```

This reproduces the core open-weight results (Tables I-IV) in approximately 6 minutes on an RTX 3080.

The script:
1. Extracts refusal direction on the extraction set (N=20)
2. Selects the best layer by AUC maximization on the validation set (yields layer 26)
3. Calibrates the threshold on the validation set (direct format)
4. Measures gap retention across all format tiers on the test set (Table I)
5. Runs format-token ablation with six progressive levels (Table II)
6. Computes calibration shift signature (Table III)
7. Performs cumulative activation patching for tool_call and system (Table IV)

Pass/fail checks compare reproduced values against the paper's expected values (with tolerance for GPU non-determinism).

For the full experiment suite including cross-model (Table V):
```bash
python experiments/run_ieee_cars_experiments.py
```

## What Is NOT Included

- **Frontier API evaluation** requires API keys for OpenAI, Google, and Anthropic. Scripts are in `experiments/`; prompt-level outcome classifications are in `frontier/prompt_outcomes.csv`. Raw response bodies are not distributed due to provider terms of service.
- **Full multi-seed fine-tuning** (5 seeds x 2 conditions x 5 formats = 50 evaluations) takes ~6 GPU-hours. Raw results are in `csv/exp_multiseed_ft.csv`; the training script is `experiments/run_multiseed_ft.py`.

## Models

All open-weight models loaded in FP16 (no quantization):

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Qwen2.5-1.5B-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B-Instruct | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| TinyLlama-1.1B-Chat | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| SmolLM2-1.7B-Instruct | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Phi-2 (control) | 2.7B | `microsoft/phi-2` |
| SmolLM2-135M-Instruct | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` (fine-tuning only) |

Frontier models (evaluated June 2026):
- GPT-4o (2024-08-06) via OpenAI API (N=50)
- Gemini 2.5 Flash via Google AI API (N=20)
- Claude Sonnet 4 via Anthropic API (N=20)

## Prompt Dataset

90 harmful + 90 matched harmless prompts across 6 categories:
- Violence/Weapons (15+15)
- Cybercrime (15+15)
- Fraud/Deception (15+15)
- Illegal Substances (15+15)
- Harassment/Abuse (15+15)
- Miscellaneous Illegal (15+15)

Three disjoint stratified splits (seed=42):
- **Extraction** (20+20): refusal direction extraction
- **Validation** (20+20): layer selection (AUC) + threshold calibration
- **Test** (50+50): all metrics reported in the paper

See `experiments/prompt_dataset.py` for the full dataset and `get_stratified_splits()` function.

## Layer Selection Protocol

1. Extract direction at ALL layers using the extraction set (difference-in-means)
2. Select best layer by maximizing AUC on the VALIDATION set
3. Calibrate threshold on the validation set (maximize balanced accuracy)
4. Report all metrics on the test set only

Result: layer 26 for Qwen2.5-1.5B-Instruct. The exact layer may vary by +/-1 due to hardware/driver non-determinism.

## Serialization

All conditions use raw tokenization:

```python
tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
```

`apply_chat_template` is never used. Format tokens in the model's vocabulary (e.g., Qwen's `<tool_call>`, `<|im_start|>`) resolve to registered token IDs. Foreign format strings are tokenized as subwords.

## Hardware

All open-weight experiments run on a single GPU with 10GB+ VRAM (tested on RTX 3080).
- Core reproduction (`reproduce.py`): ~6 minutes
- Full primary model + cross-model (`run_ieee_cars_experiments.py`): ~2 hours
- Multi-seed fine-tuning (`run_multiseed_ft.py`): ~6 GPU-hours

## Citation

```bibtex
@inproceedings{krell2026format,
  title={Format-Dependent Calibration Shifts in Refusal Representations
         of Instruction-Tuned Language Models},
  author={Krell, Jacob},
  booktitle={IEEE Cyber Awareness and Research Symposium (CARS)},
  year={2026}
}
```

## Limitations

- Mechanistic analysis limited to 1.1-3B parameter models (5 architectures)
- Frontier models (GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4) evaluated behaviorally only via API
- Fine-tuning experiments at 135M scale only (SmolLM2-135M-Instruct)
- Near-perfect within-format AUC partially reflects clearly separable prompt sets
- Causal claims are evidence-supported via activation patching, not definitively established
- GPU non-determinism may cause +/-0.5% variation in retention values across hardware

## License

MIT
