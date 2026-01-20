"""
MOCK TEST: Full Trade Lifecycle with Google Sheets
---------------------------------------------------
This test will:
1. Log an OPEN trade to Google Sheets (TradeLog tab)
2. Wait 3 seconds
3. Update the trade to CLOSED with PnL

This verifies the full GS integration.
"""

import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

def run_mock_test():
    print("🧪 MOCK TEST: Full Trade Lifecycle with Google Sheets")
    print("=" * 60)
    
    # 1. Connect directly to avoid DashboardLogger complexity
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
    client = gspread.authorize(creds)
    
    sheet_id = "1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw"
    spreadsheet = client.open_by_key(sheet_id)
    sheet_trades = spreadsheet.worksheet("TradeLog")
    
    print("✅ Google Sheets Connected")
    
    # 2. Simulate Entry
    entry_price = 25000.0
    sl = entry_price - 5  # 24995
    tp = entry_price + 40  # 25040
    
    ny_tz = pytz.timezone('America/New_York')
    ts_str = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n📥 STEP 1: Logging OPEN trade...")
    print(f"   Entry: {entry_price}, SL: {sl}, TP: {tp}")
    
    # Create row: Timestamp, PoolID, Direction, Entry, SL, TP, Status, PnL_Pts, PnL_USD, Wick, Vol, Body, ZScore
    row = [ts_str, "MOCK_TEST", "BUY", entry_price, sl, tp, "OPEN", 0, 0, 0.15, 2.0, 8.0, 1.2]
    
    try:
        sheet_trades.append_row(row)
        # Get the row index
        all_rows = sheet_trades.col_values(1)
        gs_row = len(all_rows)
        print(f"✅ Trade logged to Row {gs_row}")
    except Exception as e:
        print(f"❌ FAILED to append row: {e}")
        return
    
    # 3. Wait a bit
    print(f"\n⏳ STEP 2: Waiting 3 seconds to simulate trade duration...")
    time.sleep(3)
    
    # 4. Simulate Exit (SL Hit)
    exit_price = 24994.0  # Hit SL
    pnl_pts = exit_price - entry_price  # -6 points
    pnl_usd = pnl_pts * 2.0  # $2 per point for MNQ
    
    print(f"\n📤 STEP 3: Updating trade to CLOSED...")
    print(f"   Exit: {exit_price}, PnL: {pnl_pts} pts (${pnl_usd})")
    
    try:
        # Update columns G (Status), H (PnL_Pts), I (PnL_USD)
        cells_range = f"G{gs_row}:I{gs_row}"
        values = [["CLOSED", pnl_pts, pnl_usd]]
        sheet_trades.update(cells_range, values)
        print(f"✅ Row {gs_row} updated to CLOSED")
    except Exception as e:
        print(f"❌ FAILED to update row: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ MOCK TEST COMPLETE!")
    print(f"   Check your TradeLog sheet - Row {gs_row} should show:")
    print(f"   - Status: CLOSED")
    print(f"   - PnL_Pts: {pnl_pts}")
    print(f"   - PnL_USD: {pnl_usd}")

if __name__ == "__main__":
    run_mock_test()
