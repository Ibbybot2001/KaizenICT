import gspread
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

print("Connecting to Google Sheets...")
try:
    gc = gspread.service_account(filename="C:/Users/CEO/ICT reinforcement/service_account.json")
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw/edit")
    print(f"✅ Connected to: {sh.title}")
    
    # 1. DOWNLOAD DATA
    print("Downloading OneMinuteData...")
    ws = sh.worksheet("OneMinuteData")
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    
    print(f"Downloaded {len(df)} rows.")
    
    # 2. ANALYZE FOR GAPS
    if not df.empty and 'Timestamp' in df.columns:
        # Convert timestamp
        df['dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['dt']).sort_values('dt')
        
        # Calculate time differences
        df['diff'] = df['dt'].diff()
        
        # Find gaps > 70 seconds (allow small buffer for 60s bars)
        gaps = df[df['diff'] > timedelta(seconds=70)]
        
        if len(gaps) > 0:
            print(f"\n⚠️ FOUND {len(gaps)} DATA GAPS:")
            for idx, row in gaps.iterrows():
                prev_time = df.loc[idx-1, 'dt'] if idx-1 in df.index else "N/A"
                curr_time = row['dt']
                gap_size = row['diff']
                print(f"   Gap: {prev_time} -> {curr_time} (Duration: {gap_size})")
        else:
            print("\n✅ NO GAPS DETECTED. Data stream is continuous.")
            
        print(f"\nLatest Entry: {df.iloc[-1]['dt']}")
    else:
        print("DataFrame empty or missing Timestamp column.")

except Exception as e:
    print(f"❌ Error: {e}")
