import pandas as pd
import sys
from pathlib import Path
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

# CONFIG (From Best DNA)
DNA = {'trigger': 'Disp_60%', 'context': 'Full_US', 'sl': 15, 'tp': 50, 'manager': 'Fixed', 'be': False}
TEST_MONTHS = [2, 5, 7, 10]

def verify():
    base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
    final_pnl = 0.0
    total_trades = 0
    
    print(f"Verifying DNA: {DNA}")
    
    for m in TEST_MONTHS:
        print(f"  Testing Month {m}...")
        tick_path = base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m:02d}_clean_ticks.parquet"
        bar_path = base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m:02d}_clean_1m.parquet"
        
        df_ticks = pd.read_parquet(tick_path)
        df_bars = pd.read_parquet(bar_path)
        
        # Precompute
        df_bars['time'] = pd.to_datetime(df_bars.index)
        df_bars['hour'] = df_bars['time'].dt.hour
        df_bars['minute'] = df_bars['time'].dt.minute
        df_bars['range'] = df_bars['high'] - df_bars['low']
        
        # Trigger Logic (Disp_60%)
        mask_t = (df_bars['close'] - df_bars['open']).abs() > (df_bars['range'] * 0.6)
        
        # Context (Full_US)
        mask_c = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | ((df_bars['hour']>=10)&(df_bars['hour']<16))
        
        mask = mask_t & mask_c & (df_bars['close'] > df_bars['open'])
        signals = df_bars[mask]['time'].tolist()
        
        if not signals: continue
        
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        res = tester.backtest_signals(signals, direction=1, 
                                    stop_pts=DNA['sl'], 
                                    target_pts=DNA['tp'], 
                                    tp1_pts=None, tp1_pct=0.0, move_to_be=False)
        
        print(f"    Month {m}: PnL {res['pnl']:.2f}, Trades {res['trades']}")
        final_pnl += res['pnl']
        total_trades += res['trades']
        
    print("="*40)
    print(f"FINAL VERIFICATION PNL: {final_pnl:.2f}")
    
if __name__ == "__main__":
    verify()
