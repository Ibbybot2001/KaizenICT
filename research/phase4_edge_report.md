# Phase 4: Edge Isolation Report

## Baseline (DEEP_RETRACE, time_20 exit)
- **Events**: 30357
- **Mean R**: 3.4908
- **Win Rate**: 73.7%

---

## Analysis by Time of Day
     window  count   mean_r  win_rate   total_r
09:45-10:15   2402 2.537687  0.734804  6095.525
10:15-11:00   3719 2.654941  0.742135  9873.725
11:00-12:00   4838 2.661467  0.740182 12876.175
12:00-13:00   4800 3.073948  0.742917 14754.950
13:00-14:00   4902 3.579825  0.723990 17548.300
14:00-15:00   4770 3.759486  0.727883 17932.750
15:00-16:00   4859 5.343275  0.745421 25962.975

## Analysis by Event Type
event_type  count   mean_r  win_rate   total_r
       EQH  11574 3.307471  0.714187 38280.675
       EQL  11369 3.558699  0.760841 40458.850
       PDH   4584 3.304052  0.722731 15145.775
       PDL   2830 4.270212  0.760071 12084.700

## Analysis by Sweep Size
     sweep_type  count   mean_r  win_rate
 Micro (<3 pts)   8516 1.606194  0.725810
Macro (>=3 pts)  21841 4.225615  0.741678
       Macro Q1   5527 2.115365  0.734395
       Macro Q2   5424 2.962071  0.745575
       Macro Q3   5434 3.435582  0.742179
       Macro Q4   5456 8.406305  0.744685

## Analysis by Volatility
Empty DataFrame
Columns: []
Index: []

---

## Best Single Filters
- **hour_window**: 15:00-16:00 -> 5.3433 R (4859 trades)
- **event_type**: PDL -> 4.2702 R (2830 trades)
- **sweep_size**: Macro Q4 -> 8.4063 R (5456 trades)

---

## Next Steps
1. Cross-validate on OOS data
2. Test filter interactions
3. Define final rejection thresholds
