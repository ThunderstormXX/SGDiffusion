# EXP10: Raw SGD-Noise Alpha Sweep

This experiment interpolates between full-gradient descent and noisy SGD by
explicitly decomposing the minibatch gradient:

```text
g_batch = g_full + noise
g_alpha = g_full + alpha * noise
```

`alpha = 1` recovers ordinary SGD. Small `alpha` approaches full-gradient
descent. Large `alpha` exaggerates minibatch noise.

The sweep uses a single fixed trajectory per alpha and reports train/test loss
over 100 dataset epochs.
