# ML Battle-Testing Lab - Core Constants
# NON-NEGOTIABLE CONSTRAINTS

# === TRADING RULES ===
MIN_SL_POINTS = 10.0  # HARD RULE - every trade must have SL >= 10 points
TICK_SIZE = 0.25  # MNQ tick size
CONTRACT_MULTIPLIER = 2.0  # MNQ multiplier ($2 per point)

# === EXECUTION COSTS ===
SLIPPAGE_TICKS = 1  # Default slippage in ticks
COMMISSION_PER_CONTRACT = 1.24  # Round-trip commission per contract

# === FORBIDDEN INDICATORS ===
# These are classic technical indicators - NOT ALLOWED in this lab
FORBIDDEN_INDICATORS = frozenset([
    'RSI', 'MACD', 'SMA', 'EMA', 'WMA', 'ADX', 'ATR', 
    'BB', 'BOLLINGER', 'STOCH', 'CCI', 'MFI', 'OBV'
])

# === ALLOWED TRANSFORMS ===
# Raw OHLCV transforms (not indicators) that ARE allowed:
# - Returns (pct change)
# - Ranges (H-L)
# - Body sizes (|C-O|)
# - Z-scores of range/body
# - Volatility estimates from raw ranges
# - Learned representations (embeddings)
