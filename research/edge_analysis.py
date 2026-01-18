"""
Edge Isolation Analysis
Phase 4: Find conditions that reject losers within DEEP_RETRACE events

Target: ~90-97% rejection while preserving/amplifying expectancy
Fixed Exit: time_20 (locked, not optimized)

Filters to Test:
1. Retrace Speed (bars to deep retrace)
2. Time of Day (9:45-10:15, 10:15-11:00, etc)
3. Event Type (PDH/PDL vs EQH/EQL)
4. Sweep Size (micro vs macro)
5. Volatility Context (ATR at event)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class EdgeIsolator:
    """
    Analyzes DEEP_RETRACE events to find rejection conditions.
    """
    
    def __init__(self, outcome_df: pd.DataFrame):
        """
        Args:
            outcome_df: Phase 3 outcome matrix
        """
        self.df = outcome_df.copy()
        # Filter to DEEP_RETRACE only
        self.deep_retrace = self.df[self.df['reaction_20'] == 'DEEP_RETRACE'].copy()
        
        # Extract hour from timestamp
        self.deep_retrace['timestamp'] = pd.to_datetime(self.deep_retrace['timestamp'])
        self.deep_retrace['hour'] = self.deep_retrace['timestamp'].dt.hour
        self.deep_retrace['minute'] = self.deep_retrace['timestamp'].dt.minute
        self.deep_retrace['hour_decimal'] = self.deep_retrace['hour'] + self.deep_retrace['minute'] / 60
    
    def baseline_stats(self) -> Dict:
        """Get baseline stats for DEEP_RETRACE events."""
        return {
            'count': len(self.deep_retrace),
            'mean_r': self.deep_retrace['time_20_r'].mean(),
            'win_rate': (self.deep_retrace['time_20_result'] == 'WIN').mean(),
            'total_r': self.deep_retrace['time_20_r'].sum()
        }
    
    def analyze_by_hour_window(self) -> pd.DataFrame:
        """Analyze expectancy by time-of-day windows."""
        # Define windows
        windows = [
            ('09:45-10:15', 9.75, 10.25),
            ('10:15-11:00', 10.25, 11.0),
            ('11:00-12:00', 11.0, 12.0),
            ('12:00-13:00', 12.0, 13.0),
            ('13:00-14:00', 13.0, 14.0),
            ('14:00-15:00', 14.0, 15.0),
            ('15:00-16:00', 15.0, 16.0),
        ]
        
        results = []
        for name, start, end in windows:
            mask = (self.deep_retrace['hour_decimal'] >= start) & (self.deep_retrace['hour_decimal'] < end)
            subset = self.deep_retrace[mask]
            
            if len(subset) > 0:
                results.append({
                    'window': name,
                    'count': len(subset),
                    'mean_r': subset['time_20_r'].mean(),
                    'win_rate': (subset['time_20_result'] == 'WIN').mean(),
                    'total_r': subset['time_20_r'].sum()
                })
        
        return pd.DataFrame(results)
    
    def analyze_by_event_type(self) -> pd.DataFrame:
        """Analyze expectancy by event type."""
        results = []
        for event_type in ['EQH', 'EQL', 'PDH', 'PDL']:
            subset = self.deep_retrace[self.deep_retrace['event_type'] == event_type]
            
            if len(subset) > 0:
                results.append({
                    'event_type': event_type,
                    'count': len(subset),
                    'mean_r': subset['time_20_r'].mean(),
                    'win_rate': (subset['time_20_result'] == 'WIN').mean(),
                    'total_r': subset['time_20_r'].sum()
                })
        
        return pd.DataFrame(results)
    
    def analyze_by_sweep_size(self) -> pd.DataFrame:
        """Analyze expectancy by sweep size (micro vs macro)."""
        results = []
        
        # Micro sweeps (< 3 pts)
        micro = self.deep_retrace[self.deep_retrace['is_micro_sweep'] == True]
        if len(micro) > 0:
            results.append({
                'sweep_type': 'Micro (<3 pts)',
                'count': len(micro),
                'mean_r': micro['time_20_r'].mean(),
                'win_rate': (micro['time_20_result'] == 'WIN').mean()
            })
        
        # Macro sweeps (>= 3 pts)
        macro = self.deep_retrace[self.deep_retrace['is_micro_sweep'] == False]
        if len(macro) > 0:
            results.append({
                'sweep_type': 'Macro (>=3 pts)',
                'count': len(macro),
                'mean_r': macro['time_20_r'].mean(),
                'win_rate': (macro['time_20_result'] == 'WIN').mean()
            })
        
        # Size quartiles for macro
        if len(macro) > 0:
            macro = macro.copy()
            macro['size_quartile'] = pd.qcut(macro['sweep_size_pts'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                subset = macro[macro['size_quartile'] == q]
                if len(subset) > 0:
                    results.append({
                        'sweep_type': f'Macro {q}',
                        'count': len(subset),
                        'mean_r': subset['time_20_r'].mean(),
                        'win_rate': (subset['time_20_result'] == 'WIN').mean()
                    })
        
        return pd.DataFrame(results)
    
    def analyze_by_volatility(self) -> pd.DataFrame:
        """Analyze expectancy by ATR context."""
        results = []
        
        # ATR quartiles
        if 'atr_at_event' in self.deep_retrace.columns:
            dr = self.deep_retrace.copy()
            dr['atr_quartile'] = pd.qcut(dr['atr_at_event'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
            
            for q in ['Low', 'Med-Low', 'Med-High', 'High']:
                subset = dr[dr['atr_quartile'] == q]
                if len(subset) > 0:
                    results.append({
                        'atr_level': q,
                        'count': len(subset),
                        'mean_r': subset['time_20_r'].mean(),
                        'win_rate': (subset['time_20_result'] == 'WIN').mean()
                    })
        
        return pd.DataFrame(results)
    
    def find_best_filter_combination(self) -> Dict:
        """
        Test filter combinations and find best rejection/expectancy tradeoff.
        """
        best_filters = []
        
        # 1. Hour window analysis
        hour_df = self.analyze_by_hour_window()
        if not hour_df.empty:
            # Find windows with above-average expectancy
            avg_r = hour_df['mean_r'].mean()
            good_windows = hour_df[hour_df['mean_r'] > avg_r]
            if not good_windows.empty:
                best_hour = good_windows.loc[good_windows['mean_r'].idxmax()]
                best_filters.append({
                    'filter': 'hour_window',
                    'value': best_hour['window'],
                    'mean_r': best_hour['mean_r'],
                    'count': best_hour['count']
                })
        
        # 2. Event type analysis
        type_df = self.analyze_by_event_type()
        if not type_df.empty:
            best_type = type_df.loc[type_df['mean_r'].idxmax()]
            best_filters.append({
                'filter': 'event_type',
                'value': best_type['event_type'],
                'mean_r': best_type['mean_r'],
                'count': best_type['count']
            })
        
        # 3. Sweep size analysis
        size_df = self.analyze_by_sweep_size()
        if not size_df.empty:
            best_size = size_df.loc[size_df['mean_r'].idxmax()]
            best_filters.append({
                'filter': 'sweep_size',
                'value': best_size['sweep_type'],
                'mean_r': best_size['mean_r'],
                'count': best_size['count']
            })
        
        # 4. Volatility analysis
        vol_df = self.analyze_by_volatility()
        if not vol_df.empty:
            best_vol = vol_df.loc[vol_df['mean_r'].idxmax()]
            best_filters.append({
                'filter': 'volatility',
                'value': best_vol['atr_level'],
                'mean_r': best_vol['mean_r'],
                'count': best_vol['count']
            })
        
        return best_filters
    
    def apply_filter_stack(self, filters: List[Dict]) -> pd.DataFrame:
        """Apply a stack of filters and return filtered dataset."""
        filtered = self.deep_retrace.copy()
        
        for f in filters:
            if f['filter'] == 'hour_window':
                # Parse hour window
                window = f['value']
                parts = window.split('-')
                start = float(parts[0].replace(':', '.'))
                end = float(parts[1].replace(':', '.'))
                filtered = filtered[(filtered['hour_decimal'] >= start) & (filtered['hour_decimal'] < end)]
            
            elif f['filter'] == 'event_type':
                filtered = filtered[filtered['event_type'] == f['value']]
            
            elif f['filter'] == 'sweep_size':
                if 'Micro' in f['value']:
                    filtered = filtered[filtered['is_micro_sweep'] == True]
                else:
                    filtered = filtered[filtered['is_micro_sweep'] == False]
        
        return filtered
