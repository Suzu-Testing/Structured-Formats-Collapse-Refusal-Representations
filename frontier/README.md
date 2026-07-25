# Frontier Model Evaluation

This directory contains the evaluation script for frontier model behavioral testing.

## Requirements

```bash
pip install openai google-generativeai anthropic
```

You must set the following environment variables:
- `OPENAI_API_KEY` - for GPT-4o evaluation
- `GOOGLE_API_KEY` - for Gemini 2.5 Flash evaluation
- `ANTHROPIC_API_KEY` - for Claude Sonnet 4 evaluation

## Usage

```bash
python eval_frontier.py
```

## What This Script Tests

Each frontier model is evaluated under two conditions:
1. **Direct**: the harmful prompt is sent as a plain user message
2. **Native agentic format**: the harmful prompt is wrapped in the model's native tool/function protocol

| Model | N | Direct Condition | Format Condition |
|-------|---|-----------------|------------------|
| GPT-4o | 50 | Plain user message | tool_call with function results |
| Gemini 2.5 Flash | 20 | Plain content | function_response JSON |
| Claude Sonnet 4 | 20 | Plain user message | tool_result block |

## Parameters

All evaluations use identical generation parameters:
- `temperature = 0` (deterministic)
- `top_p = 1.0`
- `max_tokens = 512`

## Statistical Analysis

- McNemar's test (paired binary outcomes: refused vs. not-refused)
- Bonferroni correction for 3 simultaneous comparisons

## Prompt Source

The test prompts are the 50-prompt test split from `experiments/prompt_dataset.py`. GPT-4o uses all 50; Gemini and Claude use the first 20 (due to rate limits at time of evaluation).

## Reproduction Notes

Raw API response logs are not redistributed due to provider terms of service. Running this script will reproduce the evaluation from scratch. Results may vary slightly due to model updates (the paper reports results from evaluations conducted in May 2026).

Expected results (from paper):
- GPT-4o: refusal drops from ~86% to ~30% (p < 0.0001)
- Gemini 2.5 Flash: refusal drops from ~95% to ~45% (p_adj = 0.004)
- Claude Sonnet 4: no significant change (p = 1.0)
