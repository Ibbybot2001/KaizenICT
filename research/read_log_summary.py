
import re

log_path = r"c:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\liquidity_discovery\output_log_rejection_tracking.txt"

try:
    with open(log_path, 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

# Extract Stats
total_match = re.search(r"Total Setups Found: (\d+)", content)
exec_match = re.search(r"Executed: (\d+)", content)
rej_match = re.search(r"Rejected: (\d+)", content)
wr_match = re.search(r"Win Rate: ([\d.]+)%", content)
pnl_match = re.search(r"TOTAL P&L: ([+\-\d.]+)R", content)

print("="*40)
print("BACKTEST SUMMARY")
print("="*40)
if total_match: print(f"Total Setups: {total_match.group(1)}")
if exec_match: print(f"Executed:     {exec_match.group(1)}")
if rej_match:  print(f"Rejected:     {rej_match.group(1)}")
if wr_match:   print(f"Win Rate:     {wr_match.group(1)}%")
if pnl_match:  print(f"Total P&L:    {pnl_match.group(1)}")

print("\nREJECTION REASONS:")
if "Rejection Reasons:" in content:
    parts = content.split("Rejection Reasons:")
    if len(parts) > 1:
        reasons = parts[1].split("=== PERFORMANCE ===")[0]
        # Clean up whitespace
        lines = [l.strip() for l in reasons.split('\n') if l.strip()]
        for l in lines[:10]:
            print(f"  {l}")

print("\nCHARTS GENERATED:")
if "Generating charts..." in content:
    charts = content.split("Generating charts...")[1].split("Charts saved to:")[0]
    lines = [l.strip() for l in charts.split('\n') if l.strip()]
    for l in lines:
        print(f"  {l}")
