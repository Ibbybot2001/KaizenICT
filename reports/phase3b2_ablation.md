# Phase 3B.2: Ablation Grid Results

## Judgment Criteria
- ΔAUC <= -0.02: **ESSENTIAL**
- ΔAUC -0.01 to -0.02: **SUPPORTING**
- ΔAUC > -0.01: decorative

## Results by Slice


### post_displacement

| Ablation | AUC | Δ | Impact |
|----------|-----|---|--------|
| FULL | 0.496 | +0.000 | baseline |
| NO_DISPLACEMENT | 0.484 | -0.012 | SUPPORTING |
| NO_COMPRESSION | 0.489 | -0.007 | decorative |
| NO_ZONES | 0.561 | +0.065 | decorative |

### compressed

| Ablation | AUC | Δ | Impact |
|----------|-----|---|--------|
| FULL | 0.458 | +0.000 | baseline |
| NO_DISPLACEMENT | 0.455 | -0.004 | decorative |
| NO_COMPRESSION | 0.451 | -0.007 | decorative |
| NO_ZONES | 0.511 | +0.053 | decorative |

### high_context

| Ablation | AUC | Δ | Impact |
|----------|-----|---|--------|
| FULL | 0.505 | +0.000 | baseline |
| NO_DISPLACEMENT | 0.504 | -0.001 | decorative |
| NO_COMPRESSION | 0.503 | -0.002 | decorative |
| NO_ZONES | 0.502 | -0.003 | decorative |

## Summary

- Essential: none
- Supporting: {'displacement'}
