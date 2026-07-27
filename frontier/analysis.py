"""Frontier Data Analysis - IEEE CARS 2026.

Reads the frozen frontier evaluation CSVs and computes statistics.
Documents exactly which rows map to the paper's claimed N values.

KNOWN DISCREPANCIES (flagged for resolution):
1. Gemini: On-disk data (exp_frontier_scaleup.csv) shows 73.3% direct refusal vs
   56.7% formatted (p=0.18, N=30 harmful). Paper claims 95% vs 45% (N=20, p=0.002).
   These DO NOT match. Source of paper values is unresolved.
2. Claude: On-disk CSV has N=50 prompts (all harmful). Paper claims N=20.
   Subset selection criteria not documented.
3. GPT-4o: Data matches paper (86% direct, 30% system_context, N=50).
"""

import os, sys
import pandas as pd
import numpy as np

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')


def analyze_gpt4o():
    """GPT-4o analysis: N=50, 3 conditions."""
    path = os.path.join(CSV_DIR, 'exp_frontier_v3_gpt_4o.csv')
    if not os.path.exists(path):
        print('  GPT-4o CSV not found')
        return

    df = pd.read_csv(path)
    print(f'\n{"="*60}')
    print(f'  GPT-4o (exp_frontier_v3_gpt_4o.csv)')
    print(f'{"="*60}')
    print(f'  Total rows: {len(df)}')
    print(f'  Unique prompts: {df["prompt_idx"].nunique()}')
    print(f'  Conditions: {df["format"].unique().tolist()}')

    for fmt in df['format'].unique():
        sub = df[df['format'] == fmt]
        n = len(sub)
        refused = (sub['classification'] == 'refused').sum()
        rate = refused / n * 100
        print(f'  {fmt}: {refused}/{n} refused = {rate:.0f}%')

    # Paper claims: direct=86% (43/50), system_context=30% (15/50)
    direct = df[df['format'] == 'direct']
    sys_ctx = df[df['format'] == 'system_context']
    if len(direct) > 0 and len(sys_ctx) > 0:
        d_ref = (direct['classification'] == 'refused').sum()
        s_ref = (sys_ctx['classification'] == 'refused').sum()
        print(f'\n  Paper comparison:')
        print(f'    Direct: {d_ref}/50 = {d_ref/50*100:.0f}% (paper: 86%)')
        print(f'    System_context: {s_ref}/50 = {s_ref/50*100:.0f}% (paper: 30%)')

        # McNemar's test
        both = pd.merge(
            direct[['prompt_idx', 'classification']].rename(columns={'classification': 'direct_class'}),
            sys_ctx[['prompt_idx', 'classification']].rename(columns={'classification': 'sys_class'}),
            on='prompt_idx'
        )
        b = ((both['direct_class'] == 'refused') & (both['sys_class'] != 'refused')).sum()
        c = ((both['direct_class'] != 'refused') & (both['sys_class'] == 'refused')).sum()
        from scipy.stats import binom_test
        if b + c > 0:
            p = binom_test(min(b, c), b + c, 0.5) * 2
        else:
            p = 1.0
        print(f'    McNemar discordant: b={b}, c={c}, p={p:.2e}')
        print(f'    STATUS: {"MATCHES" if abs(d_ref/50*100 - 86) < 2 else "DISCREPANCY"}')


def analyze_gemini():
    """Gemini analysis: exp_frontier_scaleup.csv."""
    path = os.path.join(CSV_DIR, 'exp_frontier_scaleup.csv')
    if not os.path.exists(path):
        print('  Gemini CSV not found')
        return

    df = pd.read_csv(path)
    print(f'\n{"="*60}')
    print(f'  Gemini 2.5 Flash (exp_frontier_scaleup.csv)')
    print(f'{"="*60}')
    print(f'  Total rows: {len(df)}')
    print(f'  Columns: {df.columns.tolist()}')

    # Show what we have
    if 'prompt_type' in df.columns:
        print(f'  Prompt types: {df["prompt_type"].value_counts().to_dict()}')
    if 'condition' in df.columns:
        print(f'  Conditions: {df["condition"].unique().tolist()}')
    elif 'format' in df.columns:
        print(f'  Formats: {df["format"].unique().tolist()}')

    # Compute refusal rates from actual data
    cls_col = None
    for col in ['classification', 'outcome', 'result']:
        if col in df.columns:
            cls_col = col
            break

    if cls_col:
        # Filter to harmful prompts only
        if 'prompt_type' in df.columns:
            harmful = df[df['prompt_type'] == 'harmful']
        else:
            harmful = df

        for cond in harmful['condition'].unique() if 'condition' in harmful.columns else harmful['format'].unique():
            col_name = 'condition' if 'condition' in harmful.columns else 'format'
            sub = harmful[harmful[col_name] == cond]
            n = len(sub)
            refused = sub[cls_col].isin(['refused', 'refusal']).sum()
            rate = refused / n * 100 if n > 0 else 0
            print(f'  {cond}: {refused}/{n} refused = {rate:.1f}%')

    # Stats from frozen JSON
    stats_path = os.path.join(CSV_DIR, 'frontier_scaleup_stats.json')
    if os.path.exists(stats_path):
        import json
        with open(stats_path) as f:
            stats = json.load(f)
        print(f'\n  From frontier_scaleup_stats.json:')
        for k, v in stats.items():
            if isinstance(v, float):
                print(f'    {k}: {v:.3f}')
            else:
                print(f'    {k}: {v}')

    print(f'\n  *** DISCREPANCY FLAG ***')
    print(f'  On-disk data: ~73% direct, ~57% formatted (p=0.18, N=30 harmful)')
    print(f'  Paper claims: 95% direct, 45% formatted (p=0.002, N=20)')
    print(f'  These DO NOT match. Resolution needed from Jacob.')


def analyze_claude():
    """Claude analysis: exp_multivendor_claude.csv."""
    path = os.path.join(CSV_DIR, 'exp_multivendor_claude.csv')
    if not os.path.exists(path):
        print('  Claude CSV not found')
        return

    df = pd.read_csv(path)
    print(f'\n{"="*60}')
    print(f'  Claude Sonnet 4 (exp_multivendor_claude.csv)')
    print(f'{"="*60}')
    print(f'  Total rows: {len(df)}')
    print(f'  Unique prompts: {df["prompt_idx"].nunique()}')
    if 'format' in df.columns:
        print(f'  Conditions: {df["format"].unique().tolist()}')

    for fmt in df['format'].unique():
        sub = df[df['format'] == fmt]
        n = len(sub)
        refused = (sub['classification'] == 'refused').sum()
        rate = refused / n * 100
        print(f'  {fmt}: {refused}/{n} refused = {rate:.0f}%')

    print(f'\n  *** NOTE ***')
    print(f'  On-disk: N=50 prompts (all harmful). Paper claims N=20.')
    print(f'  Subset selection criteria not documented. Resolution needed from Jacob.')
    print(f'  Paper claims: 85% direct, 85% formatted (p=1.0)')


def main():
    print('Frontier Evaluation Data Analysis')
    print('Reads frozen CSVs and checks against paper claims.\n')

    analyze_gpt4o()
    analyze_gemini()
    analyze_claude()

    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    print('  GPT-4o: Data matches paper claims (verified)')
    print('  Gemini: DISCREPANCY - on-disk data does not match paper')
    print('  Claude: N mismatch - CSV has 50, paper claims 20')
    print('\n  See docstring for full discrepancy documentation.')


if __name__ == '__main__':
    main()
