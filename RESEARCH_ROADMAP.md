# 🧪 KaizenICT Research Roadmap: Q1 2026
**Objective**: Scale trade frequency to 3-5 trades/day in US Session without compromising edge quality.

## 🎯 Current Baseline
- **Strategy**: `TITAN_ONL_SNIPER` (Overnight Low Reclaim)
- **Status**: Deployment Ready (PF 2.15)
- **Limitation**: Low Volume (~0.5 trades/day). Depends on external pools (Daily/Weekly) being swept.

## 🔭 Research Vector 1: Intraday Liquidity Models
*Moving beyond "Pre-Market" levels to "In-Session" levels.*

### Concept A: The "Silver Bullet" (Time-Based)
**Logic**: A high-probability window for continuous volatility, largely independent of external pool sweeps.
- **Window**: 10:00 AM - 11:00 AM ET.
- **Setup**: 
  1. Any FVG (Fair Value Gap) formation.
  2. Trade the retrace into the FVG.
  3. Target: Next opposing liquidity (50/100 points).
- **Hypothesis**: This window generates 1 valid setup almost *every single day*.
- **Frequency Target**: 1-2 trades/day.

### Concept B: Opening Range Breakout (ORB) / Initial Balance
**Logic**: Using the 9:30-10:00 AM range as the liquidity pool itself.
- **Pools**: 
  - `IB_HIGH` (High of 9:30-10:00)
  - `IB_LOW` (Low of 9:30-10:00)
- **Setup**: Wait for sweep of IB High/Low -> Reversal.
- **Hypothesis**: The "AM Trend" usually reverses or accelerates after testing the initial 30m range.
- **Frequency Target**: 1 trade/day.

## 🔭 Research Vector 2: Entry Refinements
*Improving the "Trigger" to capture more moves.*

### Concept C: FVG Entry vs. Reclaim Entry
**Current**: Enter immediately when price closes back inside the level (Reclaim).
**Proposed**:
- **MSS + FVG**: Wait for the Reclaim (MSS) -> Wait for an FVG to form -> Enter on Limit Order back into the FVG.
- **Benefit**: Better Risk:Reward (tighter stop).
- **Risk**: Missed trades (price runs without retrace).

## 📅 Execution Plan

| Phase | Task | Goal |
| :--- | :--- | :--- |
| **1. Data** | Engineer `FVG` and `OpeningRange` features in `phase17_engine`. | Capability |
| **2. Search** | Run "Titan" on 10am-11am window specifically. | Discovery |
| **3. Hybrid** | Combine `ONL` Sweep + `Silver Bullet` Entry. | Synergy |

---
**Recommendation**: Start with **Phase 1 (Data Engineering)** to build the `FVG` primitives.
