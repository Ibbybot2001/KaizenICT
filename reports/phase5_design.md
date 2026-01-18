# Phase 5 Design: Continuous Edge Preservation

## 1. The Verdict from Phase 4
Phase 4 successfully falsified the hypothesis that a **discretized** Q-learning agent could capture the edge identified in Phase 3B.
- **Fact**: Edge exists (AUC > 0.55 in Phase 3B).
- **Fact**: Discretized Agent failed to trade profitably under friction (0 trades).
- **Conclusion**: The edge is thin and lives in the **continuous geometry** of the data. Binning destroys the signal-to-noise ratio required to beat friction (0.05R).

## 2. Phase 5 Objectives
1.  **Preserve Geometry**: Eliminate binning. Use raw continuous features.
2.  **Strict Discipline**: Maintain 0.05R friction. Keep "PASS" as the default state.
3.  **Probabilistic Policy**: Use models that output `P(Success)`, allowing for calibrated confidence thresholds.

## 3. Environment Upgrade: `ConceptTradingEnvV3`
### Observation Space
Type: `Box(shape=(N,))` (Continuous)
Features (Normalized where appropriate):
1.  **Regime Context**:
    - `is_post_displacement` (0/1)
    - `is_compressed` (0/1)
    - `is_hitting_zone` (0/1)
2.  **Continuous Magnitudes**:
    - `disp_zscore` (float)
    - `comp_score` (float)
    - `dist_to_zone_signed` (float, clipped/scaled)
    - `bars_since_touch` (float, scaled)
    - `bars_since_disp` (float, scaled)
    - `overlap_ratio` (float)

### Action Space
Type: `Discrete(4)` (Same as before: PASS, LONG, SHORT, EXIT)
*Note: While inputs are continuous, actions remain discrete. The Policy maps Continuous -> Probability(Discrete Action).*

## 4. Modeling Strategy: Dual Track

### Track A: Direct Policy (GBM / XGBoost) - "The Bridge"
Before trying full RL, we directly map the classification signal to the trading constraint.
- **Model**: XGBoost Classifier.
- **Target**: `1` if MaxMFE >= 1R AND MaxMFE > 2*MAE (Risk/Reward > 2). `0` otherwise.
- **Prediction**: `prob = model.predict_proba(state)`
- **Policy Rule**: 
  ```python
  if prob > CONFIDENCE_THRESHOLD:
      Action = LONG/SHORT
  else:
      Action = PASS
  ```
- **Why**: This isolates the "Prediction" capability from the "Temporal Credit Assignment" problem. It's the fastest way to confirm tradability.

### Track B: Continuous RL (PPO) - "The Optimizer"
If Track A implies potential, we use PPO to optimize the specific *timing* and *exit*.
- **Algo**: PPO (Proximal Policy Optimization).
- **Policy Net**: MLP [64, 64], Tanh activation.
- **Reward**: `PnL - Friction`.
- **Advantage**: Takes into account the *sequence* of decisions (e.g., holding through noise) which classification misses.

## 5. Experiment Plan
1.  **Implement `ConceptTradingEnvV3`** (Continuous).
2.  **Run Track A (XGBoost Policy)**:
    - Train on Phase 3B data.
    - Evaluate on Phase 4 event engine (Simulated).
    - Measure: Does `P(Win) > Thresh` yield > 0 expectancy after friction?
3.  **Run Track B (PPO)**:
    - If Track A is promising but needs better timing.
    - Train PPO on EnvV3.

## 6. Success Criteria
- **Profitability**: Net R > 0 over > 50 trades (OOS).
- **Stability**: Performance holds with 2-tick slippage simulation.
