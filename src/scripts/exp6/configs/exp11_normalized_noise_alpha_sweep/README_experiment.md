# EXP11: Normalized SGD-Noise Alpha Sweep

This experiment uses the same decomposition as EXP10, but normalizes the noise
before scaling:

```text
noise_normalized = noise * ||g_full|| / ||noise||
g_alpha = g_full + alpha * noise_normalized
```

Here `alpha = 1` means the stochastic component has the same norm as the
full-gradient component. This separates the role of direction from the raw
magnitude of minibatch noise.

The sweep uses a single fixed trajectory per alpha and reports train/test loss
over 100 dataset epochs.
