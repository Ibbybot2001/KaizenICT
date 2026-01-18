"""
Phase 2 Runner: Reaction Mapping

For each event in phase1_event_catalog.csv:
- Measure reactions at 5, 10, 20, 60 bars
- Classify reaction type (Continuation/Pullback/Retrace/Chop)
- Classify volatility behavior (Expansion/Collapse/Stable)

Output: phase2_reaction_map.csv
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from research.reaction import ReactionClassifier

DATA_PATH = "data/kaizen_1m_data_ibkr_2yr.csv"
EVENTS_PATH = "output/phases/phase1_event_catalog.csv"
OUTPUT_PATH = "output/phases/phase2_reaction_map.csv"

HORIZONS = [5, 10, 20, 60]

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

def load_events():
    """Load event catalog."""
    print(f"Loading events from {EVENTS_PATH}...")
    events = pd.read_csv(EVENTS_PATH)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    return events

def get_sweep_direction(event_type: str) -> str:
    """Determine sweep direction from event type."""
    if event_type in ['EQH', 'PDH']:
        return 'UP'  # Sweep above
    else:
        return 'DOWN'  # Sweep below

def main():
    print("=" * 60)
    print("PHASE 2: REACTION MAPPING")
    print("=" * 60)
    
    # Load data
    data = load_data()
    events = load_events()
    
    print(f"Price bars: {len(data)}")
    print(f"Events to process: {len(events)}")
    
    # Initialize classifier
    classifier = ReactionClassifier(atr_period=14)
    
    # Precompute ATR
    print("Computing ATR...")
    atr = classifier.compute_atr(data)
    
    # Process events
    print(f"\nProcessing reactions at horizons: {HORIZONS}...")
    
    results = []
    
    # Sample for speed (full run takes too long with 454k events)
    # Process all events but in batches
    sample_size = min(len(events), 50000)  # Process 50k max for initial analysis
    if len(events) > sample_size:
        print(f"Sampling {sample_size} events for initial analysis...")
        events_sample = events.sample(n=sample_size, random_state=42)
    else:
        events_sample = events
    
    for _, event in tqdm(events_sample.iterrows(), total=len(events_sample), desc="Mapping reactions"):
        bar_idx = event['bar_idx']
        event_type = event['event_type']
        sweep_direction = get_sweep_direction(event_type)
        
        # Skip if not enough future data
        if bar_idx + max(HORIZONS) >= len(data):
            continue
        
        row = {
            'timestamp': event['timestamp'],
            'event_type': event_type,
            'sweep_size_pts': event['sweep_size_pts'],
            'atr_at_event': event['atr_at_event'],
            'hour_decimal': event['hour_decimal'],
            'is_micro_sweep': event['is_micro_sweep']
        }
        
        # Measure at each horizon
        for horizon in HORIZONS:
            reaction = classifier.measure_reaction(
                data=data,
                event_idx=bar_idx,
                sweep_direction=sweep_direction,
                horizon=horizon,
                atr=atr
            )
            
            if reaction:
                row[f'reaction_{horizon}'] = reaction.reaction_type
                row[f'vol_{horizon}'] = reaction.vol_behavior
                row[f'extension_{horizon}'] = reaction.max_extension_pts
                row[f'retrace_{horizon}'] = reaction.max_retrace_pts
                row[f'net_move_{horizon}'] = reaction.net_move_pts
        
        results.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary Statistics
    print("\n" + "=" * 60)
    print("REACTION MAPPING SUMMARY")
    print("=" * 60)
    print(f"Events Processed: {len(results_df)}")
    
    for horizon in HORIZONS:
        col = f'reaction_{horizon}'
        if col in results_df.columns:
            print(f"\n--- Horizon: {horizon} bars ---")
            print(results_df[col].value_counts(normalize=True).round(3) * 100)
    
    print("\n--- Volatility Behavior (20-bar) ---")
    if 'vol_20' in results_df.columns:
        print(results_df['vol_20'].value_counts(normalize=True).round(3) * 100)
    
    print("\n--- By Event Type (20-bar reaction) ---")
    if 'reaction_20' in results_df.columns:
        print(results_df.groupby('event_type')['reaction_20'].value_counts(normalize=True).unstack().round(3) * 100)
    
    # Save
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
