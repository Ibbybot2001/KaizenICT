# Phase 5: OOS Validation Results

## FROZEN RULES (Not Optimized)
- Reaction: DEEP_RETRACE
- Sweep Size: Macro Q4 (Top Quartile)
- Time Window: 15:00-16:00
- Exit: time_20
- Direction: FADE

---

## TRAIN (Year 1)

| Metric | Value |
|--------|-------|
| Trades | 4141 |
| Mean R | 3.7315 |
| Median R | 0.7750 |
| Total R | 15452.10 |
| Win Rate | 55.0% |
| Max DD | -3261.43 R |
| Worst Month | -1516.40 R |

## TEST (Year 2 - OOS)

| Metric | Value |
|--------|-------|
| Trades | 4368 |
| Mean R | 3.0253 |
| Median R | 1.2875 |
| Total R | 13214.47 |
| Win Rate | 59.1% |
| Max DD | -3902.85 R |
| Worst Month | -1242.00 R |

---

## VERDICT

**OOS PASSED**: Test set shows positive expectancy (3.0253 R)

Edge **retained** at least 50% of in-sample performance.
