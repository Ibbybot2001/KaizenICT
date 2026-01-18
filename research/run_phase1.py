"""
Phase 1 Runner: Event Discovery

Loads data, detects all liquidity events, enriches with features.
Outputs: phase1_event_catalog.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import time

from research.events import LiquidityEventDetector
from research.event_features import EventFeatureBuilder
from research.reaction import ReactionClassifier
from research.labels import MultiPathLabeler
from research.edge_analysis import EdgeIsolator

DATA_PATH = "data/kaizen_1m_data_ibkr_2yr.csv"
EVENTS_PATH = "output/phases/phase1_event_catalog.csv"
OUTPUT_PATH = "output/phases/phase1_event_catalog.csv"

def load_data():
    """Load and prepare data."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    # Convert time column to proper DatetimeIndex
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df = df.sort_index()
    
    # Remove timezone for between_time to work
    df.index = df.index.tz_convert(None)
    
    # Filter to US session only (9:30 - 16:00 EST)
    df = df.between_time('09:30', '16:00')
    
    print(f"Loaded {len(df)} bars (US session only)")
    return df

def main():
    print("=" * 60)
    print("PHASE 1: LIQUIDITY EVENT DISCOVERY")
    print("=" * 60)
    
    # 1. Load Data
    data = load_data()
    
    # 2. Detect Events
    print("\nDetecting liquidity events...")
    detector = LiquidityEventDetector(
        equal_hl_tolerance=1.0,  # 1 point tolerance
        min_bars_between=5,
        earliest_entry=time(9, 45)
    )
    
    events = detector.detect_all(data)
    print(f"Detected {len(events)} raw events")
    
    if events.empty:
        print("No events found. Check data.")
        return
    
    # 3. Enrich with Features
    print("\nEnriching events with features...")
    feature_builder = EventFeatureBuilder(atr_period=14)
    events = feature_builder.enrich_events(events, data)
    
    # 4. Filter to tradable time (9:45+)
    events = events[events['hour_decimal'] >= 9.75]  # 9:45 = 9.75
    print(f"Events after 9:45am filter: {len(events)}")
    
    # 5. Summary Stats
    print("\n" + "=" * 60)
    print("EVENT CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total Events: {len(events)}")
    print("\nBy Type:")
    print(events['event_type'].value_counts())
    print("\nSweep Size (pts):")
    print(events['sweep_size_pts'].describe())
    print("\nSweep Size (ATR-normalized):")
    print(events['sweep_atr_normalized'].describe())
    print("\nMicro vs Macro:")
    print(events['is_micro_sweep'].value_counts())
    print("\nBy Hour:")
    events['hour'] = events['hour_decimal'].astype(int)
    print(events['hour'].value_counts().sort_index())
    
    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    events.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
