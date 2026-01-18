# PJ / ICT 2022 Model — Quant-Correct Specification

## THE OBJECTIVE (Non-Negotiable)

> **Build a quantitative execution model that mirrors PJ / ICT 2022 trading narrative while surviving realistic market friction.**

**Anything that deviates from this sequence-driven model is a FAILURE for the purposes of this project.**

---

## What We Proved (Immutable Findings)

| Component | Status | Evidence |
|-----------|--------|----------|
| **Turtle Soup (Sweep → Reclaim)** | ✅ REAL | $15+ expectancy, 11/12 months profitable |
| **FVG Retest as Entry** | ❌ DEAD | PF ~0.60 under 0.75pts friction |
| **Displacement / MSS** | ⚠️ DIAGNOSTIC | Confirms intent, not executable as trigger |
| **Time-Only Injection** | ❌ DEAD | Zero survivors in Phase 11 |
| **Draw-on-Liquidity Targets** | ✅ REAL | Opposing pool targets outperform fixed R:R |

---

## The Correct PJ / ICT Sequence (Quant-Clean)

### STEP 1 — Setup: Liquidity Sweep (UNCHANGED)
- **Concept:** Turtle Soup (C23)
- **Action:** Price MUST sweep a defined liquidity pool (PDH/PDL, Session H/L, Fractal Swing)
- **Validation:** Minimum sweep depth (wick beyond level)

### STEP 2 — Validation: Displacement (REFRAMED)
- **ICT Says:** "Displacement / MSS confirms intent"
- **Quant Reality:** Displacement is PROOF the sweep mattered, NOT an entry trigger
- **Implementation:** Use as a STATE FILTER, not a trigger
  - Strong reclaim candle body
  - Multi-bar displacement away from sweep
  - FVG **appearance** (not retest)

### STEP 3 — Entry: Market Execution (CORRECTED)
❌ **PJ Textbook (FAILS):** Wait for FVG retest, enter limit
✅ **Quant-Correct Options:**
1. **Market on Close (MOC)** of reclaim candle
2. **Delayed Market (T+1 / T+2 bars)**
3. **Breaker Retest** (structure level, not FVG)
4. **Sweep-Only Reclaim** (no FVG dependency)

> **FVG is no longer an entry location. It is an after-the-fact confirmation artifact.**

### STEP 4 — Target: Draw on Liquidity (EXPANDED)
- **Primary:** Opposing session pool (ONL → ONH)
- **Secondary:** Unmitigated liquidity (failed sweeps)
- **Tertiary:** Internal session liquidity / Composite clusters
- **NOT ALLOWED:** Fixed R:R, Fibs, arbitrary extensions

---

## Why Phase 14-15 Was "Wrong"

Phase 14-15 produced:
- A profitable long-only liquidation engine
- A portfolio pruned by expectancy law
- Asymmetric bias baked in

This is **valid discovery** but **NOT the destination**.

The user wanted:
- A sequence-driven execution model
- That mirrors how PJ / ICT traders describe trades
- With directional symmetry allowed (filtered structurally)

---

## Phase 16 Test Matrix

| Test | Question |
|------|----------|
| **16A** | Sweep → Reclaim → Market Entry (with/without displacement filter) |
| **16B** | Sweep → Reclaim → Delayed Entry (T+1, T+2) |
| **16C** | Sweep → Reclaim → Breaker Retest |
| **16D** | Directional Symmetry (Long AND Short, both filtered) |

**Success Criteria:**
- Matches PJ/ICT NARRATIVE
- Survives 0.5pts friction
- 3-5 trades/day
- Expectancy > $5/trade

---

## Implemented Files
- `strategies/research/ict_fvg_v5_stategate.py`: Original PJ logic (FVG retest — DEAD)
- `strategies/mle/phase14_pool_engine.py`: Liquidation portfolio (DISCOVERY, not DESTINATION)
- `strategies/mle/phase16_pj_engine.py`: **TO BE BUILT** (Correct PJ model)
