# Stress Test Results

## Purpose
Validate that the edge survives real-world execution conditions.
These are NOT optimizations ¡ª just fragility checks.

---

## Results

| Test | Trades | Mean R | Total R | Win Rate | Status |
|------|--------|--------|---------|----------|--------|
| BASELINE | 7546 | 4.9550 | 37390.30 | 59.8% | PASS |
| +1 pt Slippage | 7546 | 4.8550 | 36635.70 | 59.2% | PASS |
| +2 pts Slippage | 7546 | 4.7550 | 35881.10 | 58.3% | PASS |
| 1-Bar Entry Delay | 7546 | 2.0364 | 15366.58 | 53.5% | DEGRADED |
| Exit: 15 bars | 7546 | 4.6457 | 35056.30 | 59.4% | PASS |
| Exit: 25 bars | 7546 | 5.3699 | 40521.30 | 59.9% | PASS |
| 1 Trade/Day Throttle | 154 | 3.2417 | 499.23 | 53.2% | PASS |
| WORST CASE (+2 slip, delay, throttle) | 154 | -0.7003 | -107.85 | 45.5% | FAIL |

---

## Interpretation

- **PASS**: Edge survives this condition
- **DEGRADED**: Edge weakened but still positive
- **FAIL**: Edge destroyed - do NOT trade under this condition

---

## Verdict

**SOME TESTS FAILED**: WORST CASE (+2 slip, delay, throttle)
Review failed conditions before live trading.
