import gspread
import pandas as pd
import numpy as np
import os
import time
import pytz
from datetime import datetime

# Constants
SHEET_NAME_1MIN = "OneMinuteData"
SHEET_NAME_TICKS = "RawTicks"
SHEET_NAME_TRADES = "TradeLog"
CREDS_FILE = "service_account.json"

class DashboardLogger:
    def __init__(self):
        self.client = None
        self.sheet_1min = None
        self.sheet_trades = None
        
        # NaN Safety Helper
        self.sf = lambda v, default=0.0: default if pd.isna(v) or np.isinf(v) else v
        
        # Performance Tracking
        self.req_count = 0
        self.gs_tick_count = 0
        self.start_time = time.time()
        
        # Micro-Batching (Efficiency Pivot)
        self.tick_buffer = []
        self.last_flush_time = time.time()
        self.flush_interval = 1.1 # 54 req/min (Safe 1Hz rhythm with headroom)
        
        self.enabled = False
        self.ny_tz = pytz.timezone('America/New_York')
        self.connect()
        
    def get_ny_time_str(self, ts=None):
        """Standardizes any datetime or current time to NY string."""
        if ts is None:
            ts = datetime.now(self.ny_tz)
        elif isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = self.ny_tz.localize(ts)
            else:
                ts = ts.astimezone(self.ny_tz)
        return ts.strftime('%Y-%m-%d %H:%M:%S')
        
    def connect(self):
        if not os.path.exists(CREDS_FILE):
            print(f"⚠️ Google Sheets: '{CREDS_FILE}' not found. Dashboard disabled.")
            return

        try:
            self.client = gspread.service_account(filename=CREDS_FILE)
            # Try to open sheet by URL (More robust)
            SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw/edit"
            try:
                # Open the main spreadsheet
                spreadsheet = self.client.open_by_url(SPREADSHEET_URL)
                print(f"✅ Google Sheets: Connected to URL (ID: 1hcE1...)")
                
                # Get or Create Worksheets
                for name in [SHEET_NAME_1MIN, SHEET_NAME_TICKS, SHEET_NAME_TRADES]:
                    try:
                        ws = spreadsheet.worksheet(name)
                        if name == SHEET_NAME_1MIN: self.sheet_1min = ws
                        elif name == SHEET_NAME_TICKS: self.sheet_ticks = ws
                        elif name == SHEET_NAME_TRADES: self.sheet_trades = ws
                    except gspread.WorksheetNotFound:
                        ws = spreadsheet.add_worksheet(title=name, rows=1000, cols=15)
                        if name == SHEET_NAME_1MIN: self.sheet_1min = ws
                        elif name == SHEET_NAME_TICKS: self.sheet_ticks = ws
                        elif name == SHEET_NAME_TRADES: self.sheet_trades = ws

                self.enabled = True
                self.init_headers()
            except gspread.SpreadsheetNotFound:
                print(f"❌ Google Sheets: Could not access sheet by URL.")
                print("   -> Verify the bot email is added as Editor.")
        except Exception as e:
            print(f"❌ Google Sheets Connection Error: {repr(e)}")

    def init_headers(self):
        if not self.enabled: return
        try:
            # Optimize: Only check headers if the sheet has < 2 rows
            # 1. Headers for 1-Min Data
            if self.sheet_1min.row_count < 2:
                headers_1min = [
                    "Timestamp", "Price", "Action", "Details", 
                    "RelVol", "WickRatio", "BodyTicks", 
                    "DonchianHigh", "DonchianLow", "DistHigh", "DistLow",
                    "Status"
                ]
                self.sheet_1min.update('A1:L1', [headers_1min])
                
            # 2. Headers for Raw Ticks
            if self.sheet_ticks.row_count < 2:
                headers_ticks = ["Timestamp", "Price", "Size"]
                self.sheet_ticks.update('A1:C1', [headers_ticks])

            # 3. Headers for Trade Log
            if self.sheet_trades.row_count < 2:
                headers_trades = [
                    "Timestamp", "PoolID", "Direction", "Entry", 
                    "StopLoss", "TakeProfit", "Status", "PnL_Pts", "PnL_USD",
                    "Setup_WickRatio", "Setup_RelVol", "Setup_BodyTicks", "Setup_DispZscore"
                ]
                self.sheet_trades.update('A1:M1', [headers_trades])
                
        except Exception as e:
            if "429" in str(e):
                print("⚠️ Header Init: Quota reached. Skipping header check for now.")
            else:
                print(f"⚠️ Header Init Error: {e}")

    def log_min_data(self, timestamp, price, action, details, vol, wick, 
                   body_ticks=0, dh=0, dl=0, dist_h=0, dist_l=0):
        if not self.enabled: return
        
        try:
            ts_str = self.get_ny_time_str(timestamp)
            row = [
                ts_str, self.sf(price), action, details, 
                self.sf(vol), self.sf(wick), self.sf(body_ticks),
                self.sf(dh), self.sf(dl), self.sf(dist_h), self.sf(dist_l),
                "ACTIVE"
            ]
            self.sheet_1min.append_row(row)
        except Exception as e:
            print(f"⚠️ Log 1-Min Error: {repr(e)}")

    def log_tick(self, timestamp, price, size):
        if not self.enabled: return
        
        try:
            ts_str = self.get_ny_time_str(timestamp)
            row = [ts_str, self.sf(price), int(self.sf(size, 0))]
            self.tick_buffer.append(row)
            
            # Flush every 2 seconds
            if (time.time() - self.last_flush_time) >= self.flush_interval:
                self.flush_ticks()
        except Exception as e:
            print(f"⚠️ Log Tick Error: {repr(e)}")

    def flush_ticks(self):
        """Bundle and send all buffered ticks to Google Sheets."""
        if not self.enabled or not self.tick_buffer: return
        
        try:
            batch_size = len(self.tick_buffer)
            self.sheet_ticks.append_rows(self.tick_buffer)
            
            # Update Audit Counters
            self.req_count += 1
            self.gs_tick_count += batch_size
            
            # CIRCULAR BUFFER GUARD: 
            # If we've added > 10,000 ticks this session, clear the sheet to prevent overflow.
            # Local CSV still has the full history.
            if self.gs_tick_count > 10000:
                print("🧹 Google Sheets: 'RawTicks' reached 10k limit. Clearing for Rolling Window...")
                # Clear all rows except headers (A2 onwards)
                try:
                    self.sheet_ticks.batch_clear(["A2:C10000"])
                    self.gs_tick_count = 0 
                except:
                    # Fallback if sheet is smaller/larger
                    self.sheet_ticks.clear()
                    self.init_headers()
                    self.gs_tick_count = 0

            self.tick_buffer = []
            self.last_flush_time = time.time()
        except Exception as e:
            # If we hit 429 during a flush, we just keep the buffer for the next attempt
            if "429" not in str(e):
                print(f"⚠️ Flush Ticks Error: {repr(e)}")

    def get_requests_per_min(self):
        """Calculate current request frequency."""
        elapsed_mins = (time.time() - self.start_time) / 60.0
        if elapsed_mins <= 0: return 0
        return int(self.req_count / elapsed_mins)

    def log_trade(self, pool_id, direction, entry, sl, tp, status, pnl_pts=0,
                  wick=0, vol=0, body=0, zscore=0):
        """Log trade entries and return the row index for later updates."""
        if not self.enabled: return None
        
        try:
            # 1. Proactive Expansion: Ensure at least 50 spare rows
            current_rows = self.sheet_trades.row_count
            # append_row can fail if it hits the absolute grid limit (max 10M cells or sheet limit)
            # We explicitly add rows if the count is too high
            if current_rows >= 995:
                # Expand by another 1000 rows
                new_rows = current_rows + 1000
                self.sheet_trades.add_rows(1000)
                print(f"📈 Google Sheets: Expanded 'TradeLog' to {new_rows} rows.")
            
            ts_str = self.get_ny_time_str()
            
            # Multiplier: $2 per point for MNQ
            pnl_usd = float(pnl_pts) * 2.0
            
            row = [
                ts_str, pool_id, direction, float(entry),
                float(sl), float(tp), status, float(pnl_pts), pnl_usd,
                float(wick or 0), float(vol or 0), float(body or 0), float(zscore or 0)
            ]
            self.sheet_trades.append_row(row)
            
            # 2. Get the actual row index of the newly added row
            # We use the length of column A to find the last row with data
            all_rows = self.sheet_trades.col_values(1)
            new_row_idx = len(all_rows)
            
            return new_row_idx
        except Exception as e:
            print(f"⚠️ Log Trade Error: {repr(e)}")
            return None

    def update_trade_close(self, row_index, exit_price, pnl_pts):
        """Update an existing trade row with exit data."""
        if not self.enabled or not row_index: return
        
        try:
            # Column G=7, H=8, I=9
            pnl_usd = float(pnl_pts) * 2.0
            cells_range = f"G{row_index}:I{row_index}"
            values = [["CLOSED", float(pnl_pts), pnl_usd]]
            
            self.sheet_trades.update(cells_range, values)
            
            print(f"✅ Google Sheets: Row {row_index} updated to CLOSED.")
        except Exception as e:
            # If we hit the row limit here, we should try a fallback insert? 
            # But usually expansion in log_trade is enough.
            print(f"⚠️ Update Trade Error: {repr(e)}")

if __name__ == "__main__":
    # Test
    logger = DashboardLogger()
    if logger.enabled:
        logger.log_min_data(datetime.now(), 21000, "TEST", "Bot Init", 1.0, 0.0)
        logger.log_tick(datetime.now(), 21000.50, 5)
