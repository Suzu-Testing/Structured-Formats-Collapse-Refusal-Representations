# Frontier Model Evaluation

This directory contains prompt-level outcome classifications from the frontier model behavioral tests (Section V-D in the paper).

## Protocol

Each frontier model is evaluated under multiple format conditions:

- **GPT-4o**: User-message format wrapping (no system message, no native tool role). All conditions delivered as plain user messages with different structural wrapping.
- **Gemini 2.5 Flash**: Native `function_response` API structure.
- **Claude Sonnet 4**: Native `tool_result` block structure.

Classification: content-based rubric (NOT automated length threshold). Responses are classified based on semantic refusal phrases and compliance signals.

## Models and Sample Sizes

| Model | N per condition | Evaluation date |
|-------|----------------|-----------------|
| GPT-4o (2024-08-06) | 50 | June 2026 |
| Gemini 2.5 Flash | 20 (paper) / 30 (on-disk) | June 2026 |
| Claude Sonnet 4 | 20 (paper) / 50 (on-disk) | June 2026 |

## Results (from paper)

| Model | Direct refusal | Formatted refusal | p-value |
|-------|---------------|-------------------|---------|
| GPT-4o | 86% | 30% | < 10^-7 |
| Gemini 2.5 Flash | 95% | 45% | 0.002 (Holm: 0.004) |
| Claude Sonnet 4 | 85% | 85% | 1.0 |

## Data Files

- `../csv/exp_frontier_v3_gpt_4o.csv` - GPT-4o (N=50, 3 conditions: direct/tool_response/system_context)
- `../csv/exp_frontier_scaleup.csv` - Gemini 2.5 Flash (30 harmful + 30 harmless, 2 conditions: direct/function_response)
- `../csv/exp_multivendor_claude.csv` - Claude Sonnet 4 (N=50, 3 conditions: direct/tool_response/system_context)
- `../csv/exp_multivendor_gemini.csv` - FAILED run (all errors, wrong model). Do NOT use.
- `analysis.py` - Reads CSVs and computes statistics, documenting discrepancies

## Known Discrepancies

**Gemini:** On-disk data (`frontier_scaleup_stats.json`) shows 73.3% direct refusal vs 56.7% formatted (p=0.18, N=30 harmful). Paper claims 95% vs 45% (N=20, p=0.002). Source of paper values is unresolved.

**Claude:** On-disk CSV has N=50 prompts (all harmful). Paper claims N=20. Subset selection criteria not documented.

**GPT-4o:** Data matches paper claims (86% direct, 30% system_context, N=50). Verified.

## Requirements

```bash
pip install openai google-generativeai anthropic
```

Environment variables:
- `OPENAI_API_KEY` - for GPT-4o evaluation
- `GOOGLE_API_KEY` - for Gemini evaluation
- `ANTHROPIC_API_KEY` - for Claude evaluation

## Scripts

- `../experiments/frontier_evaluation.py` - Canonical frontier script (all 3 models)
- `../experiments/exp_frontier_v3.py` - GPT-4o original evaluation
- `../experiments/exp_multivendor_frontier.py` - Claude + Gemini original evaluation
- `analysis.py` - Reads frozen CSVs and verifies statistics against paper claims

## Parameters

All evaluations use:
- `temperature = 0` (deterministic)
- `max_tokens = 512` (GPT-4o), varies by model

## Statistical Analysis

- McNemar's test (paired binary outcomes: refused vs. not-refused)
- Holm-Bonferroni correction for 3 simultaneous comparisons

## Reproduction Notes

Running the evaluation scripts will reproduce from scratch. Results may vary due to model updates (the paper reports results from evaluations conducted in June 2026). Raw response bodies are not distributed due to provider terms of service.
