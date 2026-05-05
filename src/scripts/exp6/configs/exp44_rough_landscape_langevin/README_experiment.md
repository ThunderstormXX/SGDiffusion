# EXP44 rough fluctuating landscape

Theory equation tested:

Exact SGD is deterministic motion in a fluctuating minibatch landscape:

`w_{n+1}=w_n-eta (Grad_av(w_n)+Grad_stoh(a_n,b_n,w_n))`.

The standard Langevin surrogate uses centered variance only:

`dw=-Grad_av(w)dt + sqrt(eta Grad_disp(w)) dW_t`.

Expected result:

The rough sampled landscapes produce visibly strong trajectory fluctuations,
and the standard Langevin surrogate may deviate in mean, variance, or final
distribution at finite `eta`.

Interpretation:

This is a visual and moment-level diagnostic of the gap between fluctuating
landscape SGD and a Gaussian Langevin diffusion.

Limitation:

This is a one-dimensional synthetic landscape. It is meant as an illustrative
sanity check, not a neural-network result.
