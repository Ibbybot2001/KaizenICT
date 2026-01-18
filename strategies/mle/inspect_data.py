
import pandas as pd
import os

BASE_DIR = r"C:\Users\CEO\ICT reinforcement\data\GOLDEN_DATA\USTEC_2025_GOLDEN_PARQUET"
MONTH = "2025-01"
M1_FILE = f"USTEC_{MONTH}_clean_1m.parquet"
TICK_FILE = f"USTEC_{MONTH}_clean_ticks.parquet"

M1_PATH = os.path.join(BASE_DIR, M1_FILE)
TICK_PATH = os.path.join(BASE_DIR, TICK_FILE)

try:
    print(f"--- INSPECTING {M1_FILE} ---")
    df_m1 = pd.read_parquet(M1_PATH)
    print("Index Name:", df_m1.index.name)
    print("Columns:", df_m1.columns.tolist())
    print("Head:\n", df_m1.head(2))
    print("Index Type:", type(df_m1.index))
    if len(df_m1) > 0:
        print("First Index Value:", df_m1.index[0])
        if hasattr(df_m1.index[0], 'tzinfo'):
             print("Timezone:", df_m1.index[0].tzinfo)

    print(f"\n--- INSPECTING {TICK_FILE} ---")
    df_ticks = pd.read_parquet(TICK_PATH)
    print("Index Name:", df_ticks.index.name)
    print("Columns:", df_ticks.columns.tolist())
    print("Head:\n", df_ticks.head(2))
    print("Index Type:", type(df_ticks.index))
    if len(df_ticks) > 0:
        print("First Index Value:", df_ticks.index[0])
        if hasattr(df_ticks.index[0], 'tzinfo'):
             print("Timezone:", df_ticks.index[0].tzinfo)

except Exception as e:
    print(f"Inspection Failed: {e}")
