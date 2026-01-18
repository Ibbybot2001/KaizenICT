# ChronosBot: Production Engine (Phase 5)

## Overview
**ChronosBot** is a high-precision, time-based trading engine designed to capture specific volatility injections in the USTEC market.
It is built on the findings of the Phase 4 Stress Tests, which identified **Time-Based Momentum** as the only robust edge against market friction.

## Strategies
The bot exclusively trades these 4 validated concepts:
1.  **C1 (NY ORB):** 09:45 EST Breakout.
2.  **C3 (3 PM Macro):** 15:00 EST Momentum.
3.  **C8 (Last Hour):** 15:00 EST Daily Trend.
4.  **C14 (Silver Bullet AM):** 10:15 EST Momentum.

## Configuration
Settings are hardcoded in `config.py` based on the optimal "PF 13.73" backtest:
- **Stop Loss:** 10 Points
- **Take Profit:** 40 Points (1:4 R:R)
- **Max Trades:** 3 per day
- **Symbol:** USTEC

## How to Run
```bash
# Run the Bot (Live Loop)
python strategies/production/chronos_bot.py

# Run Unit Tests
python strategies/production/test_chronos.py
```

## Integration Guide (Next Steps)
The current version runs in **Mock Mode** (simulated signals).
To trade live, you must edit `chronos_bot.py`:

1.  **Incoming Data:** Connect your IBKR/WebSocket feed to populate `price_data` in `run()`.
    ```python
    # In run() loop:
    price_data = ibkr_feed.get_latest_data() 
    # Must contain: {'current': ..., 'high_0930_0944': ..., ...}
    ```
2.  **Outgoing Orders:** Connect your Order Execution API to `execute_trade()`.
    ```python
    def execute_trade(self, signal):
        ibkr_api.place_order(...)
    ```
