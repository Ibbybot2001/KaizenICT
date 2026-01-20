# 🕵️ STRATEGY DISCOVERY REPORT
**Date**: Jan 20, 2026
**Scope**: Full User Directory Scan

I have performed a deep dive into your file system to recover all "Winning Strategies" and testing artifacts. Here is the definitive list of what we found.

## 🏆 The "Holy Grail" (Best Documented Performance)
**File**: `strategies/mle/results/STRATEGY_FOUND_INTRADAY_HYBRID.md`
**Status**: Completed Research Artifact.
**Metrics**:
*   **Profit Factor**: **2.73** (Top Tier)
*   **Win Rate**: ~33% (Reward:Risk ~11:1)
*   **Drawdown**: Negligible relative to profit.

**The Edge**:
1.  **IB Strategy**:
    *   Target: **55.0 Points**
    *   Stop: **5.0 Points**
    *   Trigger: Sweep of `IB_HIGH` / `IB_LOW`
2.  **ASIA Strategy**:
    *   Target: **85.0 Points**
    *   Stop: **5.0 Points**
    *   Trigger: Sweep of `ASIA_HIGH` / `ASIA_LOW`

> [!TIP]
> This represents your most robust, long-term researched edge (12,000+ simulations).

---

## 🥈 The "Backtest Warriors" (CSV Mining Results)
**File**: `output/Phase15_TestA_Results.csv`
**Status**: Raw Mining Data.
**Top Result**:
*   **Profit Factor**: **1.76**
*   **Net Profit**: $3,412 (per unit)
*   **Config**: `TP: 78.0`, `SL: 5.0`
*   **Observation**: This confirms the "High TP" (approx 80pts) thesis seen in the Hybrid strategy.

**File**: `output/hybrid_search/hybrid_results_10k.csv`
**Status**: Massive Random Search (10,000 iterations).
**Top Result**:
*   **Profit Factor**: **1.63**
*   **Net Profit**: $5,517
*   **Edge**: `ASIA_L` triggers consistently performed best.

---

## 🥉 The "Live Candidate" (Today's Run)
**File**: `output/overnight_results/validated_strategies.json`
**Status**: **Live & Active** in `ibkr_bridge.py`.
**Current Settings**:
*   **ASIA**: TP ~49, SL ~13
*   **LONDON**: TP ~80, SL ~5
*   **Note**: These are slightly "looser" than the Holy Grail (SL 13 vs SL 5). This is typical for "Fresh" mining vs "Long Term" optimization.

---

## 🧠 ML / AI Experiments
**File**: `output/results/phase3a_results.json`
**Status**: "Failed" Experiments.
*   Most ML Models (Random Forest, XGBoost) achieved AUC ~0.52 (Random Guessing).
*   **Takeaway**: Your **Rule-Based Hybrid Strategies** (Structure + Levels) vastly outperform the complex ML models.

---

## ✅ Recommendation
1.  **Trust the Hybrid**: The "Strategies Found" Markdown file (PF 2.73) is the strongest evidence you have.
2.  **Restore if Needed**: If you feel the current bot is too loose (SL 13), we can manually inject the "Holy Grail" parameters (SL 5 / TP 55+85) into `validated_strategies.json`.
3.  **Safe Keeping**: All these files are now verified and located.

**Files are located at**:
*   `C:\Users\CEO\ICT reinforcement\strategies\mle\results\`
*   `C:\Users\CEO\ICT reinforcement\output\`
