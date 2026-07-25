# Format-Dependent Calibration Shifts in Refusal Representations of Instruction-Tuned Language Models

**IEEE CARS 2026 Submission**

## Summary

Agentic structured formats (tool calls, function execution, system prompts) induce format-dependent calibration shifts in safety representations. Models preserve relative harmfulness ordering but shift activation distributions so that calibrated safety boundaries fail to generalize.

Key results:

- Agentic-format conditions reduce refusal-direction gap by 87-97% (tool_call retains only 3-13% of baseline separation)
- Within-format AUC >= 0.995: separability is preserved, but the threshold shifts
- Format-token ablation restores the gap monotonically (Spearman rho = 1.0, p = 0.0028)
- GPT-4o refusal drops from 86% to 30% (N=50, p < 0.0001) under native tool-call framing
- Gemini 2.5 Flash drops from 95% to 45% (N=20, p_adj = 0.004) via native function_response
- Claude Sonnet 4 shows no change (p = 1.0), demonstrating the vulnerability is defensible
- Five-seed format-diverse training: refusal on tool-call format rises from 24% to 99%

## Repository Structure

```
submissions/ieee-cars-2026/       Final IEEE CARS 2026 LaTeX source
experiments/                       All experiment scripts
  prompt_dataset.py               Full 90+90 prompt set (6 categories)
  exp_safety_invariance.py        Core gap measurement + shared utilities
  exp_cross_model_transfer.py     Cross-architecture replication
  exp_format_adversarial_ft.py    Format-diverse fine-tuning (multi-seed)
  exp_activation_patching.py      Activation patching
  exp_sae_analysis.py             SAE exploratory analysis
  exp_calibration_curves.py       Calibration curve analysis
csv/                              Output data and results
  exp_multiseed_ft.csv            Training experiment raw results (5 seeds)
frontier/                         Frontier model evaluation
  eval_frontier.py                API evaluation script (GPT-4o, Gemini, Claude)
  README.md                       Reproduction instructions (requires API keys)
reproduce.py                      Reproduction pipeline (~15 min on RTX 3080)
requirements.txt                  Dependencies
```

## Quick Reproduction

```bash
pip install -r requirements.txt
python reproduce.py
```

This reproduces the core open-weight results (gap measurement, ablation, patching, calibration, cross-model) in approximately 15 minutes on an RTX 3080.

The script:
1. Selects the best layer via AUC maximization (direction from extraction set, AUC on validation set)
2. Measures gap retention across formats (Table I)
3. Runs format-token ablation with six levels (Table II)
4. Performs cumulative activation patching (Table III)
5. Demonstrates calibration/threshold transfer failure
6. Replicates the core effect on TinyLlama-1.1B

Pass/fail checks are printed against expected ranges from the paper.

## What Is NOT Included

- **Frontier API evaluation** requires API keys for OpenAI, Google, and Anthropic. The scripts that generated frontier results are in `frontier/` but raw API response logs are not redistributed due to terms of service. Results are reported in the paper with exact statistical tests (McNemar's, Bonferroni-corrected).
- **Full multi-seed fine-tuning** (5 seeds x 2 conditions x 5 formats = 50 evaluations) takes ~8 GPU-hours. Raw results are in `csv/exp_multiseed_ft.csv`; the training script is `experiments/exp_format_adversarial_ft.py`. Seeds: 42, 123, 456, 789, 1024. Model: SmolLM2-135M-Instruct.

## Models

All open-weight models auto-download from HuggingFace (~5GB total):

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Qwen2.5-1.5B-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B-Instruct (GPTQ) | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| TinyLlama-1.1B-Chat | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| SmolLM2-1.7B-Instruct | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Phi-2 | 2.7B | `microsoft/phi-2` |
| SmolLM2-135M-Instruct | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` (fine-tuning only) |

## Prompt Dataset

90 harmful + 90 matched harmless prompts across 6 categories:
- Weapons (15+15)
- Surveillance (15+15)
- Fraud (15+15)
- Social Engineering (15+15)
- Malware (15+15)
- Exploitation (15+15)

Three disjoint splits:
- **Extraction** (20+20): refusal direction extraction and layer selection
- **Validation** (20+20): threshold calibration
- **Test** (50+50): all metrics reported in the paper

See `experiments/prompt_dataset.py` for the full dataset and `get_splits()` function.

## Serialization

All conditions use raw tokenization:

```python
tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
```

`apply_chat_template` is never used. Format tokens that exist in the model's vocabulary (e.g., Qwen's `<tool_call>`, `<|im_start|>`) resolve to their registered token IDs. Foreign format strings (e.g., `<tool_call>` in TinyLlama's vocabulary) are treated as plain text and tokenized as subwords.

## Layer Selection

The paper reports layer 14 for Qwen2.5-1.5B-Instruct, selected by AUC maximization on the extraction set. The `reproduce.py` script selects the layer automatically using the extraction-set direction evaluated against validation-set AUC. The exact layer may vary slightly depending on hardware, driver versions, and float16 precision (typically layers 14-26 show the strongest effect). The older pre-release code used a hardcoded layer 27. All mid-to-late layers exhibit the format-dependent collapse; the paper reports results at the layer yielding peak extraction-set AUC.

## Hardware

All open-weight experiments run on a single GPU with 10GB+ VRAM (tested on RTX 3080). Reproduction takes approximately 15 minutes. The full experiment suite (all models, all experiments) takes approximately 6 GPU-hours.

## Citation

```bibtex
@inproceedings{krell2026format,
  title={Format-Dependent Calibration Shifts in Refusal Representations of Instruction-Tuned Language Models},
  author={Krell, Jacob},
  booktitle={IEEE Conference on Assured and Reliable Systems (CARS)},
  year={2026}
}
```

## Limitations

- Mechanistic analysis limited to 1.1-2.7B parameter models (5 architectures)
- Frontier models (GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4) evaluated behaviorally only via API
- Fine-tuning experiments at 135M scale only (SmolLM2-135M-Instruct)
- Near-perfect within-format AUC partially reflects clearly separable prompt sets
- Causal claims are evidence-supported via activation patching, not definitively established
- Legacy CSV outputs from earlier iterations are archived in `archive/csv/`

## License

MIT
