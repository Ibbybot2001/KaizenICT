import pandas as pd
import json

# Load Data
try:
    df = pd.read_csv("C:/Users/CEO/ICT reinforcement/output/overnight_search/validated_strategies.csv")
except:
    print("No CSV found.")
    exit()

# Parse Params
def parse_params(row):
    try:
        p = json.loads(row['params'])
        row['pools'] = str(p.get('pools'))
        row['direction'] = p.get('direction')
        row['session'] = str(p.get('session'))
        return row
    except:
        return row

df = df.apply(parse_params, axis=1)

# METRICS
df['TotalTrades'] = df['train_trades'] + df['test_trades']
df['TotalPnL'] = df['train_pnl'] + df['test_pnl']
df['AvgPF'] = (df['train_pf'] + df['test_pf']) / 2

# CATEGORIES

# 1. THE SNIPER (High PF, High Exp)
snipers = df[df['test_pf'] > 2.0].sort_values('test_exp', ascending=False).head(5)

# 2. THE GRINDER (High Volume, Decent PF)
grinders = df[(df['TotalTrades'] > 100) & (df['test_pf'] > 1.2)].sort_values('TotalTrades', ascending=False).head(5)

# 3. THE BALANCED (Good PF, Good Volume)
balanced = df[(df['test_pf'] > 1.5)].sort_values('TotalPnL', ascending=False).head(5)

# GENERATE MARKDOWN
with open("C:/Users/CEO/ICT reinforcement/STRATEGY_HALL_OF_FAME.md", "w", encoding='utf-8') as f:
    f.write("# 🏆 KaizenICT Strategy Hall of Fame\n")
    f.write("Confirmed winners from the Titan Overnight Search (Jan 2026).\n\n")
    
    def write_section(title, data):
        f.write(f"## {title}\n")
        if data.empty:
            f.write("_No strategies found for this category._\n\n")
            return
            
        f.write("| ID | Pools | Dir | Session | PF (Test) | Exp (Test) | Trades/Day |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in data.iterrows():
            tpd = round(r['test_trades'] / 80, 2) # Approx 4 months
            f.write(f"| {r['param_id']} | {r['pools']} | {r['direction']} | {r['session']} | **{r['test_pf']:.2f}** | **${r['test_exp']:.2f}** | {tpd} |\n")
        f.write("\n")

    write_section("🎯 The Snipers (High Precision)", snipers)
    # write_section("⚙️ The Grinders (High Volume)", grinders) 
    write_section("⚖️ The Balanced (Reliable PnL)", balanced)
    
    f.write("## 📝 deployment\n")
    f.write("To deploy a strategy, copy its ID and look up the full params in `validated_strategies.csv`.\n")

print("Hall of Fame generated.")
