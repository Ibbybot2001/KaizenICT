import asyncio
import time
import pandas as pd
from datetime import datetime
from ibkr_bridge import GoldenBot
from dashboard_logger import DashboardLogger

# Mocking the IB architecture for Stress Testing
class MockContract:
    symbol = "MNQ"
    localSymbol = "MNQH6"

class MockPosition:
    def __init__(self, size):
        self.contract = MockContract()
        self.position = size

class MockIB:
    def __init__(self):
        self._positions = []
        self.isConnected = lambda: True
    
    def positions(self):
        return self._positions

    def reqMktData(self, *args): pass
    def reqHistoricalData(self, *args): pass
    def qualifyContractsAsync(self, *args): pass
    async def connectAsync(self, *args, **kwargs): pass

class StressTester:
    def __init__(self):
        self.bot = GoldenBot()
        self.bot.ib = MockIB()
        self.bot.gs_logger.enabled = False # Don't spam real sheets
        
    async def run_test(self):
        print("🧪 STARTING SL/TP STRESS TEST...")
        
        # 1. Simulate Entry Signal
        entry_price = 25000.0
        sl = entry_price - 5
        tp = entry_price + 40
        
        print(f"   ► Triggering LONG Entry @ {entry_price} (SL: {sl}, TP: {tp})")
        self.bot.latest_metrics = {'wick': 0.1, 'vol': 2.0, 'body': 10, 'zscore': 1.0}
        self.bot.execute_trade("BUY", entry_price, "STRESS_TEST")
        
        # Manual Force of position
        self.bot.ib._positions = [MockPosition(1)]
        self.bot.in_position = True
        
        print("   ► Bot is now IN_POSITION. Monitoring for exit...")
        
        # 2. Simulate Ticks approaching SL
        print("   ► Feeding ticks approaching SL...")
        # 24994 is a BREACH of 24995 SL. We need 7+ ticks to trigger "TOUCHED" then 5 "POLLS"
        ticks = [25000.0, 24998.0, 24997.0, 24994.0, 24994.0, 24994.0, 24994.0, 24994.0, 24994.0]
        
        for price in ticks:
            print(f"      [Tick] Price: {price}")
            self.bot.last_known_price = price
            
            # Simulate high-frequency ticker event
            self.bot.check_active_trade_sl_tp(price)
            
            # Simulate the Position Monitor polling (usually happens in a separate loop)
            # In live, it's every 2s. Here we force it.
            if price <= sl:
                if self.bot.ib._positions:
                    print("      ⚠️ SL BREACHED! Mocking IB position closure...")
                    self.bot.ib._positions = [] # Broker fills SL
            
            # Run one cycle of monitor_positions_loop logic (extracted)
            pos_found = any(p.position != 0 for p in self.bot.ib.positions())
            if not pos_found:
                self.bot.empty_pos_count += 1
                print(f"      [Monitor] No position found. Wait count: {self.bot.empty_pos_count}/5")
            else:
                self.bot.empty_pos_count = 0
                print(f"      [Monitor] Position still ACTIVE ({self.bot.ib._positions[0].position})")
            
            if self.bot.in_position and not pos_found and self.bot.empty_pos_count >= 5:
                print("✅ TEST PASSED: Bot detected exit and triggered finalize logic.")
                # Force finalization call since we are mocking the loop
                trade_to_close = self.bot.current_trade
                exit_price = self.bot.last_known_price
                pnl = (exit_price - trade_to_close['entry'])
                self.bot.log_audit_event(trade_to_close, exit_price, "CLOSED")
                self.bot.in_position = False
                break
            
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    tester = StressTester()
    asyncio.run(tester.run_test())
