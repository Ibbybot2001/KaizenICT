# GPU Batch Runner Design
**Objective:** Execute backtests for 100+ concepts simultaneously using the user's local GPU (CUDA).

## Architecture
1.  **Data Loading:** Load M1 and Tick Data into RAM (Pandas), then convert essential columns (Open, High, Low, Close, Time) to `torch.Tensor` on `device='cuda'`.
2.  **Concept Vectorization:**
    *   Instead of iterating row-by-row, we define concepts as **Tensor Operations**.
    *   Example: `Signal = (Close > Open) & (Low < Low.shift(2))` becomes `(cl > op) & (lo < lo_roll)` in Torch.
3.  **Parallel Execution:**
    *   Create a Tensor of shape `(N_Concepts, N_Bars)`.
    *   Compute Entry Signals for all concepts in one pass.
    *   Compute PnL for all signals using a vectorized `SimulateTrade` kernel.
4.  **Output:** Map the GPU tensors back to the CPU for the final `mle_batch_results.csv`.

## Benefits
*   **Speed:** 100 concepts in < 5 seconds (vs 10 mins on CPU).
*   **Scale:** Can expand to 10,000 parameter combinations (Grid Search) effortlessly.

## Implementation Steps
1.  Verify PyTorch/CUDA availability (`check_gpu.py`).
2.  Create `strategies/mle/gpu_batch_runner.py`.
3.  Define the `ConceptLibrary` as a class with `@torch.jit.script` methods? Or just standard vectorized Torch.
4.  Run and Compare.
