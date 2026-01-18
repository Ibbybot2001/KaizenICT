# Phase 3B Conditional Slice Evaluation

**Question**: Do concepts beat naive vol INSIDE the states where they apply?

## Results

| Slice | N | % Data | Concept AUC | Naive AUC | Δ AUC | Winner |
|-------|---|--------|-------------|-----------|-------|--------|
| all | 29989 | 100.0% | 0.507 | 0.520 | -0.013 | NAIVE |
| near_zone | 29953 | 99.9% | 0.507 | 0.520 | -0.012 | NAIVE |
| near_fvg | 29921 | 99.8% | 0.507 | 0.519 | -0.012 | NAIVE |
| near_swing | 29827 | 99.5% | 0.504 | 0.518 | -0.014 | NAIVE |
| post_displacement | 11297 | 37.7% | 0.529 | 0.543 | -0.014 | NAIVE |
| compressed | 7168 | 23.9% | 0.506 | 0.510 | -0.004 | NAIVE |
| high_context | 16425 | 54.8% | 0.506 | 0.519 | -0.013 | NAIVE |