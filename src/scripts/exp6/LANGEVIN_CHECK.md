# Langevin baseline check

This note documents the stochastic updates used by the EXP1A/EXP1B smoke
figures after the Langevin correction.

## EXP1A toy process

The discrete SGD toy process is

```text
x_{k+1} = x_k - eta (lambda + sigma xi_k) x_k
        = (1 - eta lambda) x_k - eta sigma x_k xi_k,
xi_k ~ N(0, 1).
```

The standard additive Langevin baseline freezes the SGD noise amplitude at the
initial point:

```text
x_{k+1} = (1 - eta lambda) x_k + eta sigma |x_0| xi_k.
```

For this AR(1) process, when `|1 - eta lambda| < 1`, the variance has a finite
stationary value

```text
Var[x_infty] = eta^2 sigma^2 x_0^2 / (1 - (1 - eta lambda)^2).
```

The modified Langevin baseline uses the state-dependent SGD noise amplitude:

```text
x_{k+1} = (1 - eta lambda) x_k + eta sigma |x_k| xi_k.
```

This has the same one-step conditional variance as the multiplicative-noise SGD
toy process. The sign of the Gaussian noise is immaterial for the conditional
variance.

## EXP1B/EXP5 MLP process

For real-model experiments, the corrected Langevin baselines now use the
full-batch gradient at the current parameter vector:

```text
theta_{k+1} = theta_k - eta grad L_full(theta_k) + eta Sigma(theta_ref)^{1/2} xi_k
```

for standard Langevin, where `Sigma(theta_ref)` is a diagonal estimate of the
minibatch gradient-noise covariance at the common reference point.

The modified Langevin baseline uses the same full-gradient drift, but estimates
the diagonal minibatch gradient-noise covariance at the current parameter vector:

```text
theta_{k+1} = theta_k - eta grad L_full(theta_k) + eta Sigma(theta_k)^{1/2} xi_k.
```

The factor multiplying the noise is `eta`, not `sqrt(eta)`, because the baseline
is matched to the discrete SGD update

```text
theta_{k+1} = theta_k - eta (grad L_full(theta_k) + minibatch_noise_k).
```

Thus the one-step update covariance is `eta^2 Sigma`, matching minibatch SGD at
the same point under the diagonal Gaussian approximation.
