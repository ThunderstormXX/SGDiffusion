# EXP41 higher-order cumulant diagnostic

Theory equation tested:

The corrected Gaussian generator is a second-order approximation. Higher-order
matching generally requires higher cumulants.

Expected result:

With skewed minibatch noise, the Gaussian surrogate matches mean/variance but
misses skewness and kurtosis.

Interpretation:

This is a limitation of pure Gaussian diffusion, not a failure of the
second-order correction.

Limitation:

The non-Gaussian increment model is synthetic.
