# EXP17: Covariance Fokker--Planck Toy

## Goal

This experiment checks where the `g g^T` correction belongs.

`EXP16` compares one-step conditional likelihoods,

```math
p(w_{n+1}\mid w_n),
```

and correctly finds that the standard conditional covariance is preferred. That
is not a contradiction: the term `g(w)g(w)^T` is not a correction to the
conditional covariance. It enters the raw second moment of the increment and
therefore the marginal density / covariance evolution.

`EXP17` tests exactly this ensemble-density statement.

## Stochastic Quadratic Model

We use the stochastic quadratic loss

```math
L_n(w)=c_n+g_n^T w+\frac12 w^T H w,
```

with deterministic positive-definite Hessian `H` and stochastic linear term

```math
g_n \sim \mathcal N(0,\Sigma_g).
```

The exact discrete SGD update is

```math
w_{n+1}
=w_n-\eta(Hw_n+g_n)
=(I-\eta H)w_n-\eta g_n.
```

Let

```math
\Pi_n=\operatorname{Cov}(w_n)
```

be the ensemble covariance.

## True Discrete Covariance Recursion

For the exact SGD dynamics,

```math
\Pi_{n+1}^{\mathrm{true}}
=(I-\eta H)\Pi_n(I-\eta H)^T+\eta^2\Sigma_g.
```

Expanding the first term gives

```math
\Pi_{n+1}^{\mathrm{true}}
=
\Pi_n
-\eta(H\Pi_n+\Pi_n H)
+\eta^2 H\Pi_n H
+\eta^2\Sigma_g.
```

The term

```math
\eta^2 H\Pi_n H
```

is the discrete/raw-second-moment correction. It is the covariance-level
manifestation of the same issue as the `g(w)g(w)^T` term in the
Fokker--Planck expansion.

## Standard FP Truncation

The standard Langevin/Fokker--Planck truncation keeps only

```math
\Pi_{n+1}^{\mathrm{standard}}
=
\Pi_n
-\eta(H\Pi_n+\Pi_n H)
+\eta^2\Sigma_g.
```

It omits the finite-step drift-square term

```math
\eta^2 H\Pi_n H.
```

## Discrete / Raw-Moment FP Prediction

The discrete FP prediction keeps the full second-order raw moment:

```math
\Pi_{n+1}^{\mathrm{discrete}}
=
(I-\eta H)\Pi_n(I-\eta H)^T+\eta^2\Sigma_g.
```

For this toy model it matches the exact covariance recursion analytically.

## Metrics

The experiment generates many exact SGD trajectories and estimates

```math
\Pi_n^{\mathrm{empirical}}
```

from the ensemble. It then compares both predictions using relative Frobenius
error:

```math
E_{\mathrm{standard}}(n)
=
\frac{
\|\Pi_n^{\mathrm{empirical}}-\Pi_n^{\mathrm{standard}}\|_F
}{
\|\Pi_n^{\mathrm{empirical}}\|_F
},
```

```math
E_{\mathrm{discrete}}(n)
=
\frac{
\|\Pi_n^{\mathrm{empirical}}-\Pi_n^{\mathrm{discrete}}\|_F
}{
\|\Pi_n^{\mathrm{empirical}}\|_F
}.
```

The main reported improvement is

```math
\frac{E_{\mathrm{standard}}(n)}{E_{\mathrm{discrete}}(n)}.
```

## Expected Result

The discrete/raw-moment FP recursion should be much closer to the empirical SGD
ensemble covariance:

```math
E_{\mathrm{discrete}}(n)
\ll
E_{\mathrm{standard}}(n).
```

In the current full run:

```text
standard relative Frobenius error: 0.0631
discrete relative Frobenius error: 0.00978
improvement: 6.46x
discrete better fraction: 1.0
```

This supports the interpretation:

```math
g(w)g(w)^T
\text{ is not a transition-covariance correction,}
```

but it is necessary for accurate finite-step marginal density evolution.

## Outputs

Each run writes:

```text
metrics.json
figure_data.csv
raw_outputs.npz
figure.png
README_experiment.md
runtime.json
environment.json
```

The main figure shows:

1. covariance trace in log-log scale for empirical SGD, standard FP, and
   discrete FP;
2. relative Frobenius error to the empirical covariance.

## Commands

Smoke:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp17_covariance_fp_toy/smoke.yaml \
  --make-figure
```

Full:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp17_covariance_fp_toy/full.yaml \
  --make-figure
```

