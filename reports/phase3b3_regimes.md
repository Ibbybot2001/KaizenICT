# Phase 3B.3: Regime Modeling Results

## Approach
- **Representation Fix**: Distances capped at 20.0 (no '999')
- **Regimes**: Slices defined by state (Post-Disp, Pre-Disp Near, Compressed Approach)
- **Models**: Tested Primitives vs Zones vs Full within each regime

## Results

| Regime | N | Primitives AUC | Zones AUC | Full AUC | Best |
|--------|---|----------------|-----------|----------|------|
| Post-Displacement | - | 0.503 | 0.552 | 0.549 | Zones |
| Pre-Disp Near Zone | - | 0.501 | 0.502 | 0.501 | Zones |
| Comp & Approaching | - | 0.508 | 0.584 | 0.569 | Zones |
