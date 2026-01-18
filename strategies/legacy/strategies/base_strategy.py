import pandas as pd
from typing import Optional

class Strategy:
    def __init__(self, backtester):
        self.backtester = backtester
        # Common pre-computed data
        self.swing_highs: Optional[pd.Series] = None
        self.swing_lows: Optional[pd.Series] = None
        self.fvgs: Optional[pd.DataFrame] = None
        self.atr: Optional[pd.Series] = None

    def on_start(self, df):
        """
        Called before the simulation loop starts.
        Use this to pre-calculate indicators (Swings, FVGs, ATR).
        """
        pass

    def on_bar(self, i, bar):
        """
        Called on every bar.
        i: integer index of the current bar
        bar: the row (Series) of the dataframe
        """
        pass
