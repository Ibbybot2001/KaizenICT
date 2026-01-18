
import pandas as pd
import os

BASE_DIR = r"C:\Users\CEO\ICT reinforcement\data\GOLDEN_DATA\USTEC_2025_GOLDEN_PARQUET"
MONTH = "2025-01"
M1_FILE = f"USTEC_{MONTH}_clean_1m.parquet"
TICK_FILE = f"USTEC_{MONTH}_clean_ticks.parquet"

M1_PATH = os.path.join(BASE_DIR, M1_FILE)
TICK_PATH = os.path.join(BASE_DIR, TICK_FILE)

print("Loading Headers...")
df_m1 = pd.read_parquet(M1_PATH)
df_ticks = pd.read_parquet(TICK_PATH)

print("M1 Time Sample:", df_m1['time'].iloc[0], "Type:", type(df_m1['time'].iloc[0]))
print("Tick Time Sample:", df_ticks['time'].iloc[0], "Type:", type(df_ticks['time'].iloc[0]))

if 'time' in df_m1.columns:
    df_m1 = df_m1.set_index('time').sort_index()
if 'time' in df_ticks.columns:
    df_ticks = df_ticks.set_index('time').sort_index()

t1 = df_m1.index[0]
t2 = df_ticks.index[0]

print(f"M1 Index TZ: {t1.tzinfo}")
print(f"Tick Index TZ: {t2.tzinfo}")

# Check slice
start = t1
end = t1 + pd.Timedelta(minutes=5)
print(f"Slicing ticks from {start} to {end}...")
try:
    s = df_ticks[start:end]
    print(f"Slice result: {len(s)} ticks")
except Exception as e:
    print(f"Slice Failed: {e}")
