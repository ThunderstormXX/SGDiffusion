# EXP5 Learning-Rate Scaling Pack

This configuration uses the EXP5 runner in `analysis.mode: lr_scaling`.
It trains a shared MLP-386 reference point, computes the exact mean-Hessian
eigenbasis, and runs SGD trajectory ensembles for a grid of learning rates.

The output `bucketed_statistics.csv` reports projected variance in near-flat
and sharp Hessian eigenspace buckets. The key paper metric is whether the
late-time slope in flat directions increases with learning rate while sharp
directions remain comparatively bounded.
