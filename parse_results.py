import re
import json

log_path = "C:/Users/CEO/ICT reinforcement/output/highvol_search/highvol_search.log"

strategies = []
with open(log_path, 'r') as f:
    content = f.read()
    # Regex to capture the winning block
    # 2026... 🏆 WINNER: 1.5 trades/day | PF: 1.34 | Exp: $3.50
    # 2026...    {"pools": ...}
    
    matches = re.finditer(r'WINNER: ([\d.]+) trades/day \| PF: ([\d.]+) \| Exp: \$([\d.]+)\n.*?(\{.*?\})', content, re.DOTALL)
    
    for m in matches:
        try:
            trades_day = float(m.group(1))
            pf = float(m.group(2))
            exp = float(m.group(3))
            params = m.group(4)
            strategies.append({
                'trades_day': trades_day,
                'pf': pf,
                'exp': exp,
                'params': params
            })
        except:
            pass

print(f"Found {len(strategies)} winners so far\n")

# Sort by Trades/Day
print("=== TOP 5 BY VOLUME ===")
top_vol = sorted(strategies, key=lambda x: x['trades_day'], reverse=True)[:5]
for s in top_vol:
    print(f"Trades/Day: {s['trades_day']} | PF: {s['pf']} | Exp: ${s['exp']}")
    print(f"   {s['params'][:100]}...")
    print()

# Sort by Efficiency (PF)
print("=== TOP 5 BY EFFICIENCY ===")
top_pf = sorted(strategies, key=lambda x: x['pf'], reverse=True)[:5]
for s in top_pf:
    print(f"PF: {s['pf']} | Exp: ${s['exp']} | Trades/Day: {s['trades_day']}")
    print(f"   {s['params'][:100]}...")
    print()
