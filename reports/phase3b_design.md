# Phase 3B Design: Interaction Matrix & Ablation Plan

**Status**: PENDING USER REVIEW

---

## Core Philosophy

> Expansion = **context**, not edge  
> Zones = **conditional**, not causal  
> Direction alone = weak  
> Interaction = where edge emerges (if it exists)

---

## 1. Interaction Matrix Design

### Feature Groups

| Group | Features | Role |
|-------|----------|------|
| **Expansion Context** | disp_range_zscore, comp_score, comp_is_compressed | Energy/volatility state |
| **Structure Context** | zone_new_fvg_*, zone_new_swing_* | Context for action |
| **Liquidity Context** | liq_new_equal_highs/lows | Asymmetry markers |
| **Absolute Values** | disp_range_abs, comp_range_abs | Regime-aware amplitude |

### Interaction Terms to Test

| Interaction | Hypothesis |
|-------------|------------|
| `displacement × zone` | Impulsive move near structure = opportunity |
| `compression × zone` | Coiling near structure = pending breakout |
| `displacement × liquidity` | Sweep with force = possible reversal |
| `compression → expansion` | Coil → release pattern |

### NOT Testing

- Zone-only models (demonstrated weak standalone)
- Deep feature engineering beyond 1st-order interactions
- Any pattern requiring >10 bar lookback (stay simple)

---

## 2. Model Grid (Limited Complexity)

### Allowed Models

| Model | Config | Why |
|-------|--------|-----|
| **Gradient Boosting** | max_depth=3, n_estimators=50 | Captures nonlinear interactions, interpretable |
| **Logistic + Interactions** | L2, explicit interaction terms | Baseline for interpretability |
| **Temporal CNN** | kernel=3, 1 conv layer | Short sequence patterns only |

### Explicitly Disallowed

- Deep transformers
- LSTM > 1 layer
- Any model with >1000 parameters
- RL at this stage

---

## 3. Experiment Grid (What We Run)

### Baseline Models (Expected Behavior)

| Model ID | Features | Label | Expected |
|----------|----------|-------|----------|
| `B1_exp_only` | Expansion features only | direction | **WEAK ~53%** |
| `B2_dir_only` | Direction signal only | direction | **WEAK ~53%** |
| `B3_zone_only` | Zone features only | direction | **WEAK ~53%** |

### Interaction Models (Where Edge Should Emerge)

| Model ID | Features | Label | Test |
|----------|----------|-------|------|
| `I1_exp_zone` | Expansion × Zone | direction | Zone-gated by energy state |
| `I2_exp_liq` | Expansion × Liquidity | direction | Sweep detection |
| `I3_comp_zone` | Compression × Zone | expansion | Coil at structure |
| `I4_full` | All features | direction | Full model baseline |

---

## 4. Ablation Grid (Mandatory)

For every model that shows promising metrics, run:

| Ablation | What We Remove | What to Measure |
|----------|----------------|-----------------|
| `A1_no_disp` | All displacement features | Δ accuracy, Δ profit |
| `A2_no_comp` | All compression features | Δ accuracy, Δ profit |
| `A3_no_zones` | All zone features | Δ accuracy, Δ profit |
| `A4_no_liq` | All liquidity features | Δ accuracy, Δ profit |

### Ablation Pass Criteria

- If removing primitive causes < 1% drop → **primitive not doing work**
- If removing ANY primitive causes no drop → **model is suspicious**
- If removing displacement/compression causes large drop but zones doesn't → **expected and healthy**

---

## 5. Friction Testing (Before Any Celebration)

Every promising model must survive:

| Friction Test | Config | Pass Criteria |
|---------------|--------|---------------|
| `F1_slip` | +50% slippage (0.75 pts → 1.125 pts) | Sharpe still > 0 |
| `F2_delay` | +1 bar execution delay | Win rate stable within 5% |
| `F3_combined` | Both above | Still operating above random |

---

## 6. Monthly Stability Analysis

For every model passing friction tests:

| Check | Pass Criteria |
|-------|---------------|
| No single month > 30% of total PnL | Diversified returns |
| No month with > 20% drawdown of equity | Manageable volatility |
| Positive expectancy in ≥ 60% of months | Consistency |

If a model makes all money in 2 months → **reject regardless of aggregate metrics**

---

## 7. Execution Order

```
1. Build feature matrix with all interactions
2. Run BASELINE models (should be weak)
3. Run INTERACTION models
4. For any model > 55% accuracy:
   a. Run ablation grid
   b. Run friction tests
   c. Run monthly analysis
5. Only models surviving all 3 proceed
```

---

## 8. What We Are NOT Doing in Phase 3B

- ❌ Optimizing SL/TP
- ❌ Testing multiple horizons
- ❌ Adding new primitives
- ❌ RL or policy learning
- ❌ Declaring winners based on accuracy alone

---

## Questions for User

1. **Interaction depth**: Should I test 2-way only (A×B) or also 3-way (A×B×C)?  
   *Recommendation*: 2-way only to avoid overfitting.

2. **Target label for interaction models**: Use direction (harder) or expansion (easier)?  
   *Recommendation*: Both, with direction as primary.

3. **Train/test split**: Continue with 5-fold purged walk-forward?  
   *Recommendation*: Yes, same as Phase 3A.

---

**Awaiting approval before running any models.**
