# Trade Engine

This directory contains the core event-driven backtesting engine and trade management logic.

## Key Components

- **`event_engine.py`**: The main simulation kernel. Events are processed bar-by-bar to ensure no lookahead bias.
- **`trade.py`**: Defines `Trade`, `Order`, `Position` classes and enums.
- **`constants.py`**: System-wide constants (e.g., tick size, commissions, forbidden indicators).
- **`account.py`**: Managing account balance and margin.

## Usage

The `EventEngine` is typically initialized with a `Strategy` and a `SimulationConfig`. It iterates through the data, feeding bars to the strategy using `on_bar`.
