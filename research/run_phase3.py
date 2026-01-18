"""
Phase 3 Runner: Outcome Labeling (Multi-Path)

For each event in phase2_reaction_map.csv:
- Simulate trade outcomes under ALL exit schemas
- Record: Win/Loss, R-multiple, Duration

Output: phase3_outcome_matrix.csv

Trade Logic:
- EQH/PDH: SHORT (fade the sweep)
- EQL/PDL: LONG (fade the sweep)
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from research.labels import MultiPathLabeler

DATA_PATH = "data/kaizen_1m_data_ibkr_2yr.csv"
REACTIONS_PATH = "output/phases/phase2_reaction_map.csv"
OUTPUT_PATH = "output/phases/phase3_outcome_matrix.csv"

SCHEMAS = ['fixed_1.0r', 'fixed_1.5r', 'fixed_2.0r', 'partial_1_2', 'time_10', 'time_20', 'time_60']

def load_data():
    """Load price data."""
    print(f"Loading price data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df = df.sort_index()
    df.index = df.index.tz_convert(None)
    df = df.between_time('09:30', '16:00')
    return df

def load_reactions():
    """Load Phase 2 reactions."""
    print(f"Loading reactions from {REACTIONS_PATH}...")
    reactions = pd.read_csv(REACTIONS_PATH)
    reactions['timestamp'] = pd.to_datetime(reactions['timestamp'])
    return reactions

def main():
    print("=" * 60)
    print("PHASE 3: OUTCOME LABELING (MULTI-PATH)")
    print("=" * 60)
    
    # Load data
    data = load_data()
    reactions = load_reactions()
    
    print(f"Price bars: {len(data)}")
    print(f"Events to label: {len(reactions)}")
    
    # Initialize labeler
    labeler = MultiPathLabeler(sl_pts=10.0, friction_r=0.05)
    
    # Need to map timestamps back to bar indices
    # Create a lookup from timestamp to index
    print("Building timestamp index...")
    data_reset = data.reset_index()
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(data_reset['time'])}
    
    # Process events
    print(f"\nLabeling events under {len(SCHEMAS)} exit schemas...")
    
    results = []
    skipped = 0
    
    for _, row in tqdm(reactions.iterrows(), total=len(reactions), desc="Labeling outcomes"):
        ts = row['timestamp']
        
        # Find bar index
        if ts not in timestamp_to_idx:
            skipped += 1
            continue
        
        bar_idx = timestamp_to_idx[ts]
        
        # Skip if not enough future data
        if bar_idx + 100 >= len(data):
            skipped += 1
            continue
        
        event_type = row['event_type']
        
        # Label under all schemas
        outcomes = labeler.label_event(data, bar_idx, event_type)
        
        # Build result row
        result_row = {
            'timestamp': ts,
            'event_type': event_type,
            'reaction_20': row.get('reaction_20', 'UNKNOWN'),
            'vol_20': row.get('vol_20', 'UNKNOWN'),
            'sweep_size_pts': row.get('sweep_size_pts', 0),
            'is_micro_sweep': row.get('is_micro_sweep', False)
        }
        
        # Add outcome for each schema
        for schema, outcome in outcomes.items():
            result_row[f'{schema}_result'] = outcome.result
            result_row[f'{schema}_r'] = outcome.r_multiple
            result_row[f'{schema}_bars'] = outcome.bars_to_exit
        
        results.append(result_row)
    
    print(f"\nSkipped {skipped} events (timestamp mismatch or insufficient data)")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary Statistics
    print("\n" + "=" * 60)
    print("OUTCOME MATRIX SUMMARY")
    print("=" * 60)
    print(f"Events Labeled: {len(results_df)}")
    
    print("\n--- Expectancy by Schema (Mean R) ---")
    for schema in SCHEMAS:
        col = f'{schema}_r'
        if col in results_df.columns:
            mean_r = results_df[col].mean()
            win_rate = (results_df[f'{schema}_result'] == 'WIN').mean() * 100
            print(f"{schema}: {mean_r:+.4f} R (Win Rate: {win_rate:.1f}%)")
    
    print("\n--- Expectancy by Reaction Type (fixed_1.5r) ---")
    if 'fixed_1.5r_r' in results_df.columns:
        print(results_df.groupby('reaction_20')['fixed_1.5r_r'].agg(['mean', 'count']).round(4))
    
    print("\n--- Expectancy by Event Type (fixed_1.5r) ---")
    if 'fixed_1.5r_r' in results_df.columns:
        print(results_df.groupby('event_type')['fixed_1.5r_r'].agg(['mean', 'count']).round(4))
    
    # Save
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
