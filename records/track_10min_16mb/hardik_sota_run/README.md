# SOTA Attempt: Adam TTT + Deep Progressive Recurrence

This submission implements aggressive optimizations to break the 1.08 BPB barrier on the 10-min / 16MB track.

## Key Optimizations

1. **Adam Test-Time Training (Adam TTT)**:
   - Replaced standard SGD for TTT with Adam.
   - Faster convergence on the sparse validation tokens encountered during evaluation.
   - Complies with Rule 198 (only trains on tokens already evaluated).

2. **Parallel Residuals (Global)**:
   - Enabled Parallel Residuals (Attention and MLP computed in parallel) for ALL 11 layers.
   - Improves signal flow and training stability at high learning rates.

3. **Deep Progressive Recurrence**:
   - Implements 3 extra loops on middle layers (L3 through L6).
   - Effectively increases the virtual depth of the model to 15 layers while staying within the 11-layer physical parameter limit.
   - `enable_looping_at = 0.4` to preserve time budget for initial training.

4. **Tuned Muon Backend**:
   - Increased Muon backend steps to 6 for better orthogonality.
   - Initial `qk_gain` set to 5.5 for stronger attention signal in the early phase.

## Target Metrics
- **Target BPB**: < 1.08 (Verified run needed on 8xH100)
- **Status**: Smoke-tested on CPU; ready for GPU cluster execution.
