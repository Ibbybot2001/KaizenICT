# Phase 3A Concept Screening Report

**Label Type**: forward_return_sign
**Horizon**: 10 bars
**Threshold**: 52.0%

## Results by Primitive Family

| Family | Model | Test Acc | Train Acc | Passed |
|--------|-------|----------|-----------|--------|
| liquidity | tree | 52.4% | 51.7% | ✅ |
| displacement | tree | 52.1% | 52.2% | ✅ |
| zones | tree | 51.9% | 53.1% | ❌ |
| displacement | logistic | 51.4% | 52.9% | ❌ |
| compression | logistic | 51.3% | 52.8% | ❌ |
| liquidity | logistic | 51.3% | 52.8% | ❌ |
| compression | tree | 51.1% | 51.5% | ❌ |
| zones | logistic | 51.0% | 53.1% | ❌ |

## Passing Concepts

- liquidity
- displacement