"""
Configuration Constants for ICT Backtesting System
"""

# Strategy Parameters
TICK_SIZE = 0.25
POINT_VALUE = 20  # For MNQ (Micro E-mini Nasdaq-100), $2 per point usually, but NQ is $20. 
# MNQ is $2/point. NQ is $20/point. 
# User data is MNQ.
CONTRACT_MULTIPLIER = 2.0 

# Commission & Slippage
COMMISSION_PER_CONTRACT = 0.0  # User requested zero fees
SLIPPAGE_TICKS = 2  # Hardcoded simulator slippage (0.5 points)

# Timezone Settings
DATA_TIMEZONE = 'America/Chicago' # The input CSV seems to be Central time (IBKR default often) or user local. 
# User is in Central Standard Time based on file timestamps "17:00:00-06:00" -> CST (UTC-6).
# However, for ICT logic, we MUST convert to New York time.
TRADING_TIMEZONE = 'America/New_York'

# Killzones (NY Time)
# Killzones (Hour, Minute) in EST
KILLZONE_LONDON_OPEN = (2, 0, 5, 0)
KILLZONE_NY_OPEN     = (7, 0, 10, 0)
KILLZONE_NY_PM       = (13, 30, 16, 0)  # PM Session

# Silver Bullet Sessions (Hour, Minute) - 1 Hour windows
SILVER_BULLET_AM = (10, 0, 11, 0)
SILVER_BULLET_PM = (14, 0, 15, 0)

def is_in_killzone(timestamp):
    """
    Checks if the given timestamp is within any defined Killzone.
    """
    t = timestamp.time()
    t_minutes = t.hour * 60 + t.minute

    def check(zone_tuple):
        sh, sm, eh, em = zone_tuple
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        return start_min <= t_minutes <= end_min

    # Check Killzones
    if check(KILLZONE_LONDON_OPEN):
        return True
    if check(KILLZONE_NY_OPEN):
        return True
    if check(KILLZONE_NY_PM):
        return True
    
    return False

def is_in_silver_bullet(timestamp):
    """
    Checks if time is within Silver Bullet windows.
    """
    t = timestamp.time()
    t_minutes = t.hour * 60 + t.minute

    def check(zone_tuple):
        sh, sm, eh, em = zone_tuple
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        return start_min <= t_minutes <= end_min

    if check(SILVER_BULLET_AM):
        return True
    if check(SILVER_BULLET_PM):
        return True
    return False

# ICT Core Defaults
SWING_LOOKBACK = 3 # Standard 3-bar swing
FVG_DISPLACEMENT_FACTOR = 1.0 # ATR multiplier for displacement validation
