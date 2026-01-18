
import pandas as pd
import time
from strategies.mle.backtest import BacktestConfig, run_backtest, load_data

def run_batch_tests():
    print("====================================")
    print("MLE BATCH RUNNER - 20 CONCEPTS")
    print("====================================")
    
    # 1. Load Data ONCE
    try:
        df_m1, df_ticks = load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Define Concepts (Grid Search)
    # --- Batch C: High-Edge Filters (On Market Mode Base) ---
    
    # Base: Market Mode (Concept 2 - Realistic)
    base_cfg = BacktestConfig(
        execution_mode='MARKET',
        entry_strategy='MARKET_CLOSE',
        latency_ms=500,
        fvg_tolerance_ticks=0
    )
    
    # 11. Killzone Only (09:30-11:00 NY)
    cfg_11 = BacktestConfig(
        execution_mode='MARKET',
        entry_strategy='MARKET_CLOSE',
        latency_ms=500,
        filter_killzone=True
    )
    configs.append(cfg_11)
    
    # 12. Trend Align (EMA Filter)
    cfg_12 = BacktestConfig(
        execution_mode='MARKET',
        entry_strategy='MARKET_CLOSE',
        latency_ms=500,
        filter_trend_ema=True
    )
    configs.append(cfg_12)
    
    # 13. Dealing Range (Discount/Premium)
    cfg_13 = BacktestConfig(
        execution_mode='MARKET',
        entry_strategy='MARKET_CLOSE',
        latency_ms=500,
        filter_dealing_range=True
    )
    configs.append(cfg_13)
    
    # Combined: Killzone + Trend
    cfg_combo = BacktestConfig(
        execution_mode='MARKET',
        entry_strategy='MARKET_CLOSE',
        latency_ms=500,
        filter_killzone=True,
        filter_trend_ema=True
    )
    configs.append(cfg_combo)
    
    
    results = []
    
    for i, cfg in enumerate(configs):
        concept_id = i + 1
        print(f"\n--- Running Concept {concept_id} ---")
        print(f"Config: {cfg}")
        
        start_t = time.time()
        res = run_backtest(cfg, df_m1, df_ticks)
        elapsed = time.time() - start_t
        
        res['concept_id'] = concept_id
        res['config'] = str(cfg)
        res['runtime'] = f"{elapsed:.1f}s"
        results.append(res)
        
    # 3. Report
    print("\n====================================")
    print("FINAL BATCH RESULTS")
    print("====================================")
    print(f"{'ID':<5} | {'Trades':<8} | {'WR%':<8} | {'PF':<8} | {'PnL (Ticks)':<12} | {'Config'}")
    
    for r in results:
        print(f"{r['concept_id']:<5} | {r['trades']:<8} | {r['wr']:<8.2f} | {r['pf']:<8.2f} | {r['pnl']:<12.1f} | ...")
        
    # Save to CSV
    pd.DataFrame(results).to_csv("output/mle_batch_results.csv")
    print("\nResults saved to output/mle_batch_results.csv")

if __name__ == "__main__":
    run_batch_tests()
