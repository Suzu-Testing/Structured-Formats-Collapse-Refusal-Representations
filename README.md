# Format-Dependent Calibration Shifts in Refusal Representations of Instruction-Tuned Language Models

**IEEE Cyber Awareness and Research Symposium (CARS) 2026**

> Paper source: `submissions/ieee-cars-2026/` (IEEEtran format)

## Summary

Agentic structured formats (tool calls, function execution, system prompts) induce format-dependent calibration shifts in safety representations. Models preserve relative harmfulness ordering but shift activation distributions so that calibrated safety boundaries fail to generalize.

Key results (Qwen2.5-1.5B-Instruct, layer 26, N=50 test pairs):

- Tier C formats (native vocab tokens) retain only 3.5-8.3% of the direct-format refusal gap (91-97% reduction)
- Tier B formats (generic structure) retain 17-45%
- Within-format AUC remains >= 0.978: separability is preserved but the threshold shifts
- Removing format tokens generally restores the gap; bracket and key-value controls do not form a uniquely ordered sequence
- System format: Layer-0 attention-only patching restores 81.7%, MLP-only restores 42.6%
- Tool_call Layer-0 attention patching: 50.7% (not negligible); distributed across layers 0-18
- Behavioral framing stress tests: GPT-4o 86%->30% (N=50, p < 10^-7), Claude 34%->12% (N=50, p=0.019)
- Five-seed format-diverse training (matched volume, 410 seq/condition): tool-call refusal rises from 46% to 95%

## Repository Structure

```
submissions/ieee-cars-2026/              IEEE paper source (main.tex, main.pdf)
experiments/                             All experiment scripts
  prompt_dataset.py                     Full 90+90 prompt set (6 categories, stratified splits)
  run_ieee_cars_experiments.py          Canonical script producing Tables I-V
  run_qwen3b.py                         Qwen2.5-3B cross-model (FP16)
  run_multiseed_ft_v2.py                Matched-volume fine-tuning (Table IX, 410 seq/cond)
  run_multiseed_ft.py                   (deprecated: unmatched-volume version)
  verify_component_patching.py          Component-level analysis
  verify_meanpool.py                    Mean-pooled readout comparison
  verify_stats.py                       Statistical verification
  exp_safety_invariance.py              Core shared utilities
  exp_mechanistic_utils.py              Mechanistic analysis utilities
  exp_path_patching.py                  Component-level patching (attention/MLP)
  exp_attention_head_routing.py         Layer-0 attention head analysis
  exp_sae_intervention.py              SAE (160 vectors, 19 features, exploratory)
  frontier_evaluation.py                Canonical frontier script (GPT-4o, Gemini, Claude)
  run_frontier_final_v2.py             Gemini native protocol (google.genai SDK)
  run_frontier_final_v3.py             Gemini with agentic prompts
  analyze_claude.py                    Claude stats (automated heuristic)
  exp_frontier_v3.py                    GPT-4o behavioral evaluation (N=50)
  exp_multivendor_frontier.py           Claude + Gemini evaluation
  exp_cross_arch_ablation.py            Cross-architecture ablation
csv/                                    Frozen experiment results
  ieee_cars_stratified_results.json     Primary model results (Tables I-IV)
  exp_multiseed_ft_v2.csv               Matched-volume training results (5 seeds x 2 cond x 5 fmt)
  exp_multiseed_ft.csv                  (deprecated: unmatched-volume training)
  exp_cross_arch_ablation.csv           Cross-architecture ablation
  exp_cross_arch_layerwise.csv          Layer-by-layer gap retention
  exp_path_patching.csv                 Component-level patching
  exp_attention_head_routing.csv        Attention head routing
  exp_sae_intervention.csv              SAE intervention
  exp_frontier_v3_gpt_4o.csv            GPT-4o outcomes (N=50, verified)
  exp_multivendor_claude.csv           Claude Sonnet 4.6 outcomes (N=50, verified)
  exp_frontier_final_gemini_native.csv Gemini native protocol (N=20, verified)
  exp_frontier_scaleup.csv             Gemini older run (N=30+30, supplementary)
frontier/                               Frontier model evaluation summary
  analysis.py                          Read CSVs and verify statistics vs paper
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
- **Full multi-seed fine-tuning** (5 seeds x 2 conditions x 5 formats, matched volume at 410 sequences per condition, float32 with mixed-precision) takes ~6 GPU-hours. Raw results are in `csv/exp_multiseed_ft_v2.csv`; the training script is `experiments/run_multiseed_ft_v2.py`.

## Models

All open-weight mechanistic models loaded in FP16 (no quantization). Fine-tuning uses float32 with mixed-precision.

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Qwen2.5-1.5B-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B-Instruct | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| TinyLlama-1.1B-Chat | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| SmolLM2-1.7B-Instruct | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Phi-2 (control) | 2.7B | `microsoft/phi-2` |
| SmolLM2-135M-Instruct | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` (fine-tuning only) |

Frontier models (behavioral framing stress tests, June-July 2026):
- GPT-4o (`gpt-4o` rolling alias) via OpenAI API (N=50, authorized-context user-message framing)
- Claude Sonnet 4.6 (`claude-sonnet-4-6`, active pinned) via Anthropic API (N=50, tool-output text in user msg)
- Gemini 2.5 Flash (`gemini-2.5-flash` stable ID) via google.genai SDK (N=20, native FunctionResponse + instruction preamble)

Cross-model (Table V): Qwen2.5-3B uses N=30 pairs; all others use N=50.

## Prompt Dataset

90 harmful + 90 matched harmless prompts across 6 categories:
- Violence/weapons (15+15)
- Cybercrime (15+15)
- Fraud/deception (15+15)
- Illegal substances (15+15)
- Harassment/abuse (15+15)
- Miscellaneous illegal activity (15+15)

Three disjoint stratified splits (seed=42):
- **Extraction** (20+20): refusal direction extraction
- **Validation** (20+20): layer selection (AUC) + threshold calibration
- **Test** (50+50): all metrics reported in the paper

See `experiments/prompt_dataset.py` for the full dataset and `get_stratified_splits()` function.

## Layer Selection Protocol

1. Extract direction at ALL layers using the extraction set (difference-in-means)
2. Select best layer by maximizing AUC on the VALIDATION set (NOT extraction set)
3. Calibrate threshold on the validation set (maximize balanced accuracy)
4. Report all metrics on the test set only

Result: layer 26 for Qwen2.5-1.5B-Instruct. The exact layer may vary by +/-1 due to hardware/driver non-determinism.

## Tier Taxonomy

- **Tier A**: Direct (unformatted baseline)
- **Tier B**: Generic structural formats (JSON, XML, YAML, OpenAI-style non-native)
- **Tier C**: Agentic-format conditions with model-native special tokens (system `<|im_start|>`, tool_call `<tool_call>`) plus MCP JSON-RPC (included as an agentic protocol condition, not as evidence of Qwen training exposure)

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
- Multi-seed fine-tuning (`run_multiseed_ft_v2.py`): ~6 GPU-hours

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
- Frontier evaluations are behavioral framing stress tests (conditions conflate format with instructional framing); not causal proof that format alone produced the effect
- Frontier classification uses automated heuristic (length < 80 chars + keyword matching), not manual review
- Fine-tuning experiments at 135M scale only (SmolLM2-135M-Instruct)
- Near-perfect within-format AUC partially reflects clearly separable prompt sets
- Causal claims are evidence-supported via activation patching, not definitively established
- Ablation is not strictly monotonic (bracket delimiters > key-value pairs in retention)
- SAE analysis is exploratory (160 vectors, 19 features, single training run)
- GPU non-determinism may cause +/-0.5% variation in retention values across hardware

## License

MIT
