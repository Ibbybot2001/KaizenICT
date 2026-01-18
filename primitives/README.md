# Concept Primitives

This directory contains "atomic" market concepts implemented as standalone detectors. These are the building blocks of ICT strategies.

## Philosophy

Primitives are **Facts**, not **Signals**. They describe *what happened*, not *what to do*.

## Components

- **`displacement.py`**: Detects energetic price moves (Fair Value Gaps precursors).
- **`zones.py`**: Manages structural zones (FVGs, Order Blocks, Liquidity Pools) as objects with lifecycle (Active, Mitigated, Inverted).
- **`structure.py`**: Detects Swing Highs/Lows and Market Structure Shifts (MSS).
- **`liquidity.py`**: Tracks resting liquidity (Equal Highs/Lows).
- **`compression.py`**: Detects low-volatility coiling behavior.

## Testing

Run tests in `tests/` to verify no-lookahead compliance and logic correctness.
