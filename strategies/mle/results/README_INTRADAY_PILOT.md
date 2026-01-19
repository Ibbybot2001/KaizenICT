# 📉 Research Log: Intraday Pilot (Opening Range)
**Date**: Jan 19, 2026
**Target**: Initial Balance Breakout (09:30 - 10:00 range)

## 🔬 Hypothesis
Can we generate 3-5 trades/day by treating the **Opening Range (IB)** as a liquidity pool to be swept and reclaimed?

## 📊 Results
- **Combos Tested**: 36 (14-Core Parallel Search)
- **Period**: Jan 2025 - Dec 2025
- **Volume**: High (~10 trades/day).
- **Profit Factor**: **0.60 - 0.90** (Negative Expectancy).

## 💡 Conclusion
**Raw Sweeps of the Opening Range are NOT profitable.**
The 9:30-10:00 AM range is too noisy. Price frequently sweeps both sides without reversing meaningfully, or whipsaws (fake reclaim).

## 🔄 Pivot Strategy
We must filter these setups using **Time** and **Formation**:
1.  **Silver Bullet Window**: Restrict to 10:00 AM - 11:00 AM.
2.  **FVG Requirement**: Entry *must* occur inside a Fair Value Gap, not just a line cross.

**Next Step**: Implement `FVG` logic in `phase17_engine.py`.
