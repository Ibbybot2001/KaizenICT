# Strategies

This directory contains trading strategies.

## Structure

- **`legacy/`**: Contains the original Python implementations (Model 2022, Silver Bullet, OTE, etc.) ported from the old structure. These use the `strategies.legacy` namespace.
- **`research/`**: Contains new, research-grade strategies (e.g., `ict_fvg_strategy.py`, `refined_strategy_v2.py`) often used in ML pipelines.

## Usage

Legacy strategies are typically run via `main.py` in the root. 
Research strategies are often run via specific runners in `ml/` or `research/` directories.
