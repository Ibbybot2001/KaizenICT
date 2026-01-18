
import pandas as pd
import numpy as np
from ml_lab.ml.feature_builder import FeatureBuilder

DATA_PATH = "ml_lab/data/kaizen_1m_data_ibkr_2yr.csv"

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print("Columns:", df.columns)
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    elif 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time')
        
    df.columns = [c.lower() for c in df.columns]
    print("Index dtype:", df.index.dtype)
    print(df.head())
    return df

def main():
    data = load_data()
    data = data.iloc[:1000]
    
    print("\nRunning FeatureBuilder on subset...")
    fb = FeatureBuilder()
    try:
        prim = fb.build_feature_matrix(data)
        print("FeatureBuilder success!")
        print(prim.head())
    except Exception as e:
        print(f"FeatureBuilder failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
