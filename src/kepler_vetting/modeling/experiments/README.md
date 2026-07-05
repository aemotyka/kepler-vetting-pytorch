# Archived modeling experiments

This package contains trainer scripts for secondary, failed, or diagnostic
modeling branches from the Kepler vetting project.

The final selected model is `fused_tabular_local_cnn` with the 80/10/10
grouped split and fixed 0.5 threshold.

These archived modules are intentionally kept instead of deleted so the
experiment history remains reproducible, while the top-level modeling
package stays focused on the final pipeline and core baselines.

Archived branches include:

- soft-label fused local
- candidate-weighted fused local
- three-class fused local
- rescue stacker
- selective rescue rule
- two-stage rescue/veto gate
- older fused local-feature/residual/multiscale/stacked-score variants,
  when present
