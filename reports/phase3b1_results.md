# Phase 3B.1: Representation Fix Results

## Zone-Relative Features
- dist_to_nearest_zone (signed)
- dist_to_nearest_fvg (signed)
- dist_to_nearest_swing (signed)
- zone_age, inside_zone, approaching

## Results

| Slice | N | Concept AUC | Naive AUC | Δ | Winner |
|-------|---|-------------|-----------|---|--------|
| all | 24989 | 0.512 | 0.485 | +0.027 | CONCEPT |
| near_zone_3pt | 24160 | 0.512 | 0.490 | +0.023 | CONCEPT |
| near_zone_5pt | 24511 | 0.512 | 0.486 | +0.025 | CONCEPT |
| near_fvg_3pt | 23737 | 0.499 | 0.487 | +0.012 | CONCEPT |
| inside_zone | 21264 | 0.498 | 0.482 | +0.016 | CONCEPT |
| approaching_zone | 14108 | 0.521 | 0.502 | +0.019 | CONCEPT |
| post_displacement | 2085 | 0.537 | 0.458 | +0.079 | CONCEPT |
| compressed | 5710 | 0.530 | 0.482 | +0.048 | CONCEPT |
| high_context | 7554 | 0.516 | 0.485 | +0.031 | CONCEPT |
