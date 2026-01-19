# 🏆 Strategy Found: Titan Intraday Hybrid (PF 1.88)
**Date**: Jan 19, 2026
**Target**: Liquidity Sweep + FVG Reversal (Intraday)

## 🔬 The Setup
This strategy combines the high frequency of Intraday Liquidity Pools (Opening Range & Overnight) with the structural confirmation of the "Silver Bullet" (FVG).

1.  **Pools**: `IB_HIGH`, `IB_LOW`, `ONH`, `ONL`.
2.  **Trigger**: Price sweeps a pool level.
3.  **Confirmation**: Price reverses and forms a **Fair Value Gap (FVG)**.
4.  **Entry**: Limit Buy/Sell at the FVG Proximal Line.
5.  **Exits**: TP 30-50 pts, SL 5 pts past swing.

## 📊 Performance (Jan-Dec 2025)
- **Profit Factor**: **1.88** (Best Case)
- **Win Rate**: ~55-60% (Estimated from PF/RR)
- **Trades**: 172 (~1 per trading day)
- **Best Settings**:
    - `tp_target`: 30 pts
    - `sl_buffer`: 5.0 pts

## 💡 Why it Works
Blind sweeps (Pilot 1) failed (PF 0.6) because price often continues expanding.
Blind FVGs (Pilot 2) failed (PF 0.99) because many FVGs are just noise.
**Hybrid** works because:
- The **Sweep** identifies the "Trap" (stops triggered).
- The **FVG** confirms the "Smart Money Reversal".

## 🚀 Next Steps
1.  Deploy to `production_backtest_audit.py` to verify with full fees/slippage.
2.  Add to Live Engine.
