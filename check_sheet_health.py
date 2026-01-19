import gspread
import pandas as pd
from datetime import datetime

print("Connecting to Google Sheets...")
try:
    gc = gspread.service_account(filename="C:/Users/CEO/ICT reinforcement/service_account.json")
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw/edit")
    
    print(f"✅ Connected to: {sh.title}")
    
    # Check 1-Min Data
    ws_1min = sh.worksheet("OneMinuteData")
    rows_1min = ws_1min.get_all_values()
    print(f"\n[OneMinuteData] Rows: {len(rows_1min)}")
    if len(rows_1min) > 1:
        print(f"Last Row: {rows_1min[-1]}")
    
    # Check Raw Ticks
    ws_ticks = sh.worksheet("RawTicks")
    rows_ticks = ws_ticks.get_all_values()
    print(f"\n[RawTicks] Rows: {len(rows_ticks)}")
    if len(rows_ticks) > 1:
        print(f"Last Row: {rows_ticks[-1]}")
    
    # Check Trade Log
    ws_trades = sh.worksheet("TradeLog")
    rows_trades = ws_trades.get_all_values()
    print(f"\n[TradeLog] Rows: {len(rows_trades)}")

except Exception as e:
    print(f"❌ Error: {e}")
