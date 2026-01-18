# ML Lab

This directory contains the machine learning pipeline for discovering and validating trading concepts.

## Pipeline

1.  **Phase 1 (Discovery)**: Identifies raw events in the market.
2.  **Phase 2 (Reaction)**: Maps price reactions to events.
3.  **Phase 3 (Interaction)**: Trains ML models to predict outcomes based on primitive interactions.
4.  **Phase 3A (Screening)**: Tests individual primitives for predictive power.
5.  **Phase 3B (Validation)**: Validates models with strict constraints.
6.  **Policy Learning**: Reinforcement learning environments (`envs/`) for training agents.

## Key Files

- **`feature_builder.py`**: Constructs feature vectors from primitives.
- **`label_generator.py`**: Creates target labels (e.g., forward return) for training.
- **`interaction_runner.py`**: Orchestrates ML experiments.
