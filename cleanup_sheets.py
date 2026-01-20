"""
CLEANUP: Delete old rows from Google Sheets to stay under 10M cell limit.
This keeps only the last N rows of each sheet.
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials

SHEET_ID = "1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw"
KEEP_ROWS = {
    "OneMinuteData": 5000,
    "RawTicks": 50000,  # Keep last ~14 hours of ticks
    "TradeLog": 500     # Keep last 500 trades
}

def cleanup():
    print("🧹 CLEANUP: Reducing Google Sheets to stay under cell limit...")
    
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)
    
    for tab_name, keep_count in KEEP_ROWS.items():
        try:
            sheet = spreadsheet.worksheet(tab_name)
            current_rows = sheet.row_count
            
            if current_rows <= keep_count + 1:  # +1 for header
                print(f"   {tab_name}: {current_rows} rows (OK, no cleanup needed)")
                continue
            
            # Delete rows from 2 to (current - keep_count)
            delete_count = current_rows - keep_count - 1
            print(f"   {tab_name}: {current_rows} rows. Deleting {delete_count} old rows...")
            
            # Delete in chunks to avoid timeout
            while delete_count > 0:
                chunk = min(1000, delete_count)
                sheet.delete_rows(2, 2 + chunk - 1)  # Keep header (row 1)
                delete_count -= chunk
                print(f"      Deleted {chunk} rows...")
            
            print(f"   ✅ {tab_name}: Now has {sheet.row_count} rows")
            
        except Exception as e:
            print(f"   ❌ {tab_name}: Error - {e}")

if __name__ == "__main__":
    cleanup()
