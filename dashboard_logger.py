import gspread
import pandas as pd
from datetime import datetime
import os
import time

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
        
        # Performance Tracking
        self.req_count = 0
        self.gs_tick_count = 0
        self.start_time = time.time()
        
        # Micro-Batching (Efficiency Pivot)
        self.tick_buffer = []
        self.last_flush_time = time.time()
        self.flush_interval = 1.1 # 54 req/min (Safe 1Hz rhythm with headroom)
        
        self.enabled = False
        self.connect()
        
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
            ts_str = str(timestamp)
            row = [
                ts_str, float(price), action, details, 
                float(vol), float(wick), float(body_ticks),
                float(dh), float(dl), float(dist_h), float(dist_l),
                "ACTIVE"
            ]
            self.sheet_1min.append_row(row)
        except Exception as e:
            print(f"⚠️ Log 1-Min Error: {repr(e)}")

    def log_tick(self, timestamp, price, size):
        if not self.enabled: return
        
        try:
            ts_str = str(timestamp)
            row = [ts_str, float(price), int(size)]
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
        """Log trade entries and exits with max description capturing."""
        if not self.enabled: return
        
        try:
            ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Multiplier: $2 per point for MNQ
            pnl_usd = float(pnl_pts) * 2.0
            
            row = [
                ts_str, pool_id, direction, float(entry),
                float(sl), float(tp), status, float(pnl_pts), pnl_usd,
                float(wick or 0), float(vol or 0), float(body or 0), float(zscore or 0)
            ]
            self.sheet_trades.append_row(row)
        except Exception as e:
            print(f"⚠️ Log Trade Error: {repr(e)}")

if __name__ == "__main__":
    # Test
    logger = DashboardLogger()
    if logger.enabled:
        logger.log_min_data(datetime.now(), 21000, "TEST", "Bot Init", 1.0, 0.0)
        logger.log_tick(datetime.now(), 21000.50, 5)
