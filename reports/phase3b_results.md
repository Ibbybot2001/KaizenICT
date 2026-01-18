# Phase 3B Interaction Discovery Report

**Label**: forward_return_sign
**Horizon**: 10 bars
**AUC Threshold**: 0.54
**Max AUC Gap**: 0.03
**Baseline Accuracy**: 50.8%

## Results

| Model | Type | Acc | AUC | Gap | Fold Std | Status |
|-------|------|-----|-----|-----|----------|--------|
| C1_naive_vol | control | 53.7% | 0.531 | -0.008 | 0.035 | FAIL: AUC 0.531 < 0. |
| I2_exp_liq | interaction | 53.6% | 0.522 | 0.003 | 0.024 | FAIL: AUC 0.522 < 0. |
| B_displacement | baseline | 53.7% | 0.522 | -0.001 | 0.028 | FAIL: AUC 0.522 < 0. |
| I1_exp_zone | interaction | 53.7% | 0.516 | 0.008 | 0.025 | FAIL: AUC 0.516 < 0. |
| B_compression | baseline | 53.6% | 0.516 | 0.001 | 0.027 | FAIL: AUC 0.516 < 0. |
| I4_full | interaction | 53.6% | 0.516 | 0.010 | 0.024 | FAIL: AUC 0.516 < 0. |
| I3_comp_zone | interaction | 53.6% | 0.511 | 0.010 | 0.022 | FAIL: AUC 0.511 < 0. |
| B_liquidity | baseline | 53.6% | 0.502 | 0.012 | 0.027 | FAIL: AUC 0.502 < 0. |
| C0_do_nothing | control | 50.8% | 0.500 | 0.000 | 0.000 | CONTROL: baseline re |
| B_zones | baseline | 53.6% | 0.491 | 0.021 | 0.010 | FAIL: AUC 0.491 < 0. |

## Promising Models for Ablation

*No models met promotion criteria.*