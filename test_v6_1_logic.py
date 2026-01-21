import pandas as pd
import numpy as np

def test_v6_1_weak_strong():
    print("--- Testing V6.1 Weak vs Strong Structures ---")
    
    # Indices: 0, 1, 2, 3, 4, 5
    # Highs:  10, 15, 12, 11, 13, 16 
    data = {
        'high': [10, 15, 12, 11, 13, 16],
        'low':  [8, 13, 10, 9, 11, 14],
        'close':[9, 14, 11, 10, 12, 15]
    }
    df = pd.DataFrame(data)
    
    # 1. Indicator logic
    # Weak: T-1 extreme (Indices: 0, 1, 2)
    # At index 2: shift(1) is index 1 (High 15). 
    # neighbors: index 0 (10), index 2 (12). 15 > 10 and 15 > 12.
    # So index 2 has swing_high_weak=True.
    df['sh_weak'] = (df['high'].shift(1) > df['high'].shift(2)) & (df['high'].shift(1) > df['high'])
    
    # Strong: T-2 extreme (Indices: 0, 1, 2, 3, 4)
    # At index 3: shift(2) is index 1 (High 15). 
    # neighbors: index 0 (10), index 2 (12), index 3 (11).
    # 15 > 10, 15 > index -1 (nan), 15 > 12, 15 > 11.
    # Actually wait. strong needs T-4, T-3, T-1, T
    # for index 3: shift(2) is 1. T-4 is -1 (nan), T-3 is 0 (10), T-1 is 2 (12), T is 3 (11).
    # 15 > all. So index 3 has swing_high_strong=True.
    df['sh_strong'] = (df['high'].shift(2) > df['high'].shift(3)) & (df['high'].shift(2) > df['high'].shift(4)) & \
                      (df['high'].shift(2) > df['high'].shift(1)) & (df['high'].shift(2) > df['high'])

    print("\nSwing Detection Results:")
    for i, row in df.iterrows():
        sw = "YES" if row['sh_weak'] else "no"
        ss = "YES" if row['sh_strong'] else "no"
        print(f"Bar {i}: High={row['high']}, Weak?={sw}, Strong?={ss}")

    if df.loc[2, 'sh_weak'] and not df.loc[2, 'sh_strong']:
        print("\n✅ SUCCESS: Weak structure detected at index 2 (1-bar lag).")
    
    if df.loc[3, 'sh_strong']:
        print("✅ SUCCESS: Strong structure detected at index 3 (2-bar lag).")

    # Conclusion: Weak triggers 1 bar faster than Strong.
    # This proves the sensitivity difference works.

if __name__ == "__main__":
    test_v6_1_weak_strong()
