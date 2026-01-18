import time
import json
import os
import sys
from datetime import datetime

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_state():
    try:
        with open("live_dashboard.json", "r") as f:
            return json.load(f)
    except:
        return None

def main():
    print("Initializing Dashboard...")
    while True:
        state = load_state()
        if state:
            clear()
            # HEADER
            print("=========================================================")
            print("       PJ/ICT EXECUTION DASHBOARD (LIVE)        ")
            print("=========================================================")
            print(f" TIME: {state.get('time', '--:--:--')}   |   STATUS: {state.get('status', 'UNK')}")
            print("---------------------------------------------------------")
            
            # KPI
            pnl = state.get('pnl', 0.0)
            pnl_str = f"${pnl:,.2f}"
            print(f"\n CURRENT PNL:      {pnl_str}")
            print(f" DAILY TRADES:     {state.get('trades', 0)}")
            print(f" LAST PRICE:       {state.get('last_price', 0):.2f}")
            
            # POOLS
            print("\n [ACTIVE POOLS]")
            pools = state.get('pools', [])
            if pools:
                print(f" > {', '.join(pools)}")
            else:
                print(" > (None)")
                
            print("\n=========================================================")
            print(" Monitoring... (Press Ctrl+C to Exit)")
        
        else:
            print("Waiting for Engine Heartbeat...", end='\r')
            
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
