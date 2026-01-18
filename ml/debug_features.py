
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import numpy as np
from ml_lab.ml.zone_relative import ZoneRelativeBuilder

def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df

def main():
    data = load_data().iloc[:1000]
    builder = ZoneRelativeBuilder()
    feats = builder.build_features(data)
    
    print("\nFeature Columns:")
    print(feats.columns)
    
    print("\nDtypes:")
    print(feats.dtypes)
    
    print("\nNumeric Columns (per select_dtypes):")
    numeric = feats.select_dtypes(include=[np.number])
    print(numeric.columns)
    
    print("\nValue Range (dist_to_nearest_zone):")
    print(f"Min: {feats['dist_to_nearest_zone'].min()}")
    print(f"Max: {feats['dist_to_nearest_zone'].max()}")
    print(f"NaNs: {feats['dist_to_nearest_zone'].isna().sum()}")
    
    if feats['dist_to_nearest_zone'].max() > 50:
        print("FAIL: Max > 50 (Capping failed?)")
    elif feats['dist_to_nearest_zone'].isna().sum() > 0:
        print("FAIL: NaNs present")
    else:
        print("PASS: Capping looks correct")
        
if __name__ == '__main__':
    main()
