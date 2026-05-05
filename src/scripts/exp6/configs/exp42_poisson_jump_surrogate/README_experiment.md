# EXP42 Poisson jump surrogate

Theory motivation:

Higher-order matching may require jump or Poisson-type processes rather than a
pure Gaussian diffusion.

Expected result:

For skewed minibatch increments, a simple compound-Poisson surrogate should fit
the increment distribution better than a Gaussian surrogate.

Interpretation:

This is exploratory. It motivates future non-Gaussian generators but is not a
main claim of the paper.

Limitation:

The surrogate is one-dimensional and moment-matched heuristically.
