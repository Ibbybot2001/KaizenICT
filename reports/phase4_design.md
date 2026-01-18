# Phase 4: Edge Thickening & Winner Selection

## Objective
Convert the "Rational Null Policy" (Phase 3C) into a "Selective Active Policy" by sharpening the agent's view of the market and refining the reward signal. We are **not** inventing new edges; we are increasing the resolution of the existing ones to find the pockets of profitability that survive friction.

## Core Philosophy
> "Edge Thickening, not Edge Invention."

## Audit Improvements (Pre-Requisites)
Before applying the levers, we must fix the resolution issues flagged in Phase 3C:
1.  **Explicit Time-Since-Event**: `is_displacement` is too broad. We need `bars_since_displacement` (0..5) to allow the agent to learn timing (e.g., immediate follow-through vs. delayed reaction).
2.  **Explicit Null Handling**: Define "Null Regime" behavior strictly. **Decision**: Forced Exit. If the regime expires or invalidates, the trade is scratched. The agent is a "Regime Specialist"; it has no authority outside the regime.
3.  **Explicit Trade Counting**: Track trades via state transitions (`position 0 -> 1`), not reward heuristics.

## The Three Levers

### Lever 1: Sharpen State Resolution (The Lens)
We will increase the granularity of the state space *only* within the valid regimes.

*   **Post-Displacement Regime**:
    *   `disp_intensity`: [2-3 sigma, 3-5 sigma, >5 sigma] (Existing)
    *   **[NEW]** `bars_since_disp`: [1, 2, 3, 4, 5+] (Crucial for momentum decay learning)
    *   **[NEW]** `dist_to_zone_fine`: [-5..-2, -2..0, 0..2, 2..5] (Focus on the "interaction zone")

*   **Compressed & Approaching Regime**:
    *   `comp_score`: [0.5-0.8, >0.8] (High compression focus)
    *   **[NEW]** `approaching_velocity`: Fast vs Slow approach (derived from speed primitive)? *Consensus: Keep it simple first. Use `bars_since_touch` or similar if needed, or just relying on `dist` change.*
    *   **[NEW]** `dist_to_zone_fine`: As above.

### Lever 2: Outcome-Aware Reward Shaping (The Compass)
The current reward is purely PnL-based (lagging and sparse). We will add *dense* but *aligned* signals to guide exploration, without removing the friction penalties.

*   **Outcome Bonus**:
    *   If Trade hits 1R: +0.2R bonus (Reward "reaching the target" even if subsequent trail stops out).
    *   If Trade hits 2R: +0.5R bonus.
    *   *Purpose*: Distinguish "good ideas that failed execution" from "bad ideas".
*   **Safety Valve (Retained)**:
    *   Entry Cost: -0.1R (Strict filter).
    *   Time Cost: -0.01R (Urgency).

### Lever 3: Policy Restriction (The Guardrails)
*   **Confirmation Requirement**:
    *   Can we enforce "Candle Color Match"? E.g. Only Enter Long if current bar (t) close > open?
    *   *Implementation*: Add `candle_color` to state OR hard-code action mask. **Decision**: Hard-code mask. Action `ENTER_LONG` only valid if `close > open`.

## Experiment Plan

### Experiment 4.1: "Resolution"
*   **Hypothesis**: The agent passed because it couldn't distinguish "fresh" displacement from "stale" displacement.
*   **Change**: Update `ConceptTradingEnv` with `bars_since` and `fine_dist` features.
*   **Goal**: See if Agent finds profitable sub-pockets (e.g. "Enter Long on Bar 2 if Dist < 2").

### Experiment 4.2: "Shaping"
*   **Hypothesis**: The sparse PnL reward makes it hard to link "Action A" to "Result B" across time gaps.
*   **Change**: Add Outcome Bonuses.
*   **Goal**: Faster convergence, potentially discovering deeper targets.

## Evaluation Criteria
1.  **Selectivity**: Must remain low frequency (trades < Random).
2.  **Profitability**: Positive Expectancy (> 0 R) after all costs.
3.  **Stability**: Performance holds across 2-year dataset (or train/test split).
