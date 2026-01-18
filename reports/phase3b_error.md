============================================================
PHASE 3B: INTERACTION DISCOVERY
============================================================
Loaded 699726 bars
Using 50000 bars for experiments
Running Phase 3B experiments...

--- CONTROLS ---
C0 do-nothing: acc=0.508
C1 naive vol:  acc=0.537, AUC=0.531

--- BASELINES ---
B_displacement: acc=0.537, AUC=0.522 
B_compression: acc=0.536, AUC=0.516 
B_zones: acc=0.536, AUC=0.491 
B_liquidity: acc=0.536, AUC=0.502 

--- INTERACTIONS ---
I1 exp×zone: acc=0.537, AUC=0.516
I2 exp×liq:  acc=0.536, AUC=0.522
I3 comp×zone: acc=0.536, AUC=0.511
I4 full:     acc=0.536, AUC=0.516

============================================================
SUMMARY
============================================================

Promising models: 0

Report saved to: C:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\reports\phase3b_results.md
JSON saved to: C:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\runs\phase3b_results.json
