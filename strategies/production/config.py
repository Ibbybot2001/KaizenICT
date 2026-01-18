
# ChronosBot Configuration
# Validated by Phase 4 Stress Test (Jan 2025)

# --- Risk Management ---
# Rigid R:R derived from PF 13.73 performance.
STOP_LOSS_PTS = 10.0
TAKE_PROFIT_PTS = 40.0
MAX_DAILY_TRADES = 3
MAX_DAILY_LOSS_PTS = 30.0 # 3 losses * 10 pts

# --- Instrument ---
SYMBOL = "USTEC" # Nasdaq 100
TICK_SIZE = 0.25

# --- Strategy Constraints ---
# 0 = No Commission (Market Frictions Only Mode)
COMMISSION_PER_TRADE = 0.0 
SLIPPAGE_TOLERANCE_PTS = 1.0 # Allow 1pt slippage on entry

# --- Time Zones ---
TIMEZONE = "America/New_York"

# --- Trigger Schedules (HH:MM) ---
# Cron-style triggers for the "Time-Based Momentum" edge.
SCHEDULE = {
    "C1_NY_ORB": "09:45",
    "C3_3PM_MACRO": "15:00",
    "C8_LAST_HOUR_MOMENTUM": "15:00", # Merged trigger with C3? Logic differs.
    "C14_SILVER_BULLET": "10:15" # Breakout of 10:00 Open
}
