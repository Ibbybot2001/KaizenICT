# Phase 3C: Policy Learning Design — The "Fenced Garden"

**Principle**: The agent is a **decision optimizer**, not a signal discoverer. It optimizes *timing* and *direction* within strictly defined edge regimes.

## 1. State Machine (The Fence)

The agent is **blind** unless in a valid regime.

### State A: Post-Displacement (Momentum)
*   **Trigger**:
    *   `is_displacement == True`
    *   `bars_since_disp <= 5` (Temporal locality)
    *   `abs(dist_to_nearest_zone) <= 20` (Spatial relevance)
*   **Features (Discretized)**:
    *   `disp_zscore` Bins: [2-3, 3-5, >5]
    *   `dist_to_zone` Bins: [-20..-10, -10..-5, -5..-2, -2..2, 2..5, 5..10, 10..20]
*   **Hypothesis**: Momentum continuation until Zone intersection.

### State B: Comp & Approaching (Structure)
*   **Trigger**:
    *   `is_compressed == True`
    *   `approaching == True`
    *   `abs(dist_to_nearest_zone) <= 15` (Acting INTO structure)
*   **Features (Discretized)**:
    *   `comp_score` Bins: [0-0.5, 0.5-0.8, >0.8]
    *   `dist_to_zone` Bins: (Same as above)
*   **Hypothesis**: Expansion imminent; Zone dictates direction.

### State C: Null (No-Trade)
*   **Trigger**: All other conditions.
*   **Action**: Forced `PASS`.

---

## 2. Action Space (Constrained)

Discrete actions. `SL >= 10` is enforced by the Environment.

| Action | ID | Logic |
| :--- | :--- | :--- |
| **PASS / HOLD** | 0 | Do nothing. Maintain current state. |
| **ENTER_LONG** | 1 | Open Long. Cost: -0.1R. |
| **ENTER_SHORT** | 2 | Open Short. Cost: -0.1R. |
| **EXIT** | 3 | Close position. |

---

## 3. Reward Function (Math)

$$ R_t = R_{pnl} - C_{friction} - P_{time} - C_{entry} $$

Where:
*   $R_{pnl}$: Realized R-multiple. Win(+2R), Loss(-1R).
*   $C_{friction}$: Comm + Slip (approx 0.1R).
*   $P_{time}$: -0.01R per bar held.
*   **$C_{entry}$**: **-0.1R fixed penalty** on Entry. (Safety Valve: Forces selectivity).

---

## 4. Learning Algorithm

Start simple.

**Method**: **Tabular Q-Learning** (discretized features) or **PPO** (if continuous features needed).
*   Given feature dimentionality is low (~3-4 per state), **Tabular/Tile Coding** is preferred for transparency.

**Training Protocol**:
1.  **Phase 3C.2**: Build Gym Env.
2.  **Phase 3C.3**: Run Heuristic Baseline (e.g., "If Post-Disp + Zone > 0, Enter").
3.  **Phase 3C.4**: Train Agent.
4.  **Eval**: Compare Agent vs Baseline.

---

## 5. Execution Plan

1.  **Define Environment**: `ml_lab/ml/policy/concept_env.py`
2.  **Define Agent**: `ml_lab/ml/policy/agent.py`
3.  **Train**: `ml_lab/ml/run_policy_train.py`
