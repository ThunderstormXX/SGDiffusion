# Experiment Section Roadmap

Current strong blocks:

- EXP2 / EXP2 NanoGPT: quantitative validation of Eq. key.
- EXP5 lr_scaling: learning-rate dependence in flat directions.
- EXP3 sampling: with-vs-without replacement exists, but needs a clearer variance-ratio plot.

Recommended upgrades without overwriting old results:

- EXP13 should become the main Langevin falsification figure. The MLP version is
  for development; the paper version should use NanoGPT or a larger CNN/ResNet
  and show flat/sharp variance dynamics for SGD vs Langevin surrogates.
- EXP14 should replace ad hoc lr-scaling discussion by explicit eta and
  batch-size scaling summaries.
- EXP15 should be used as the reviewer-facing check of the assumption
  Var(G_i) approximately tracks positive Hessian curvature.
- EXP4 should be replaced or extended by a true Lanczos/HVP large-model sanity
  check.
- Optional future EXP17: Newton/preconditioned-step heavy-tail diagnostic.

