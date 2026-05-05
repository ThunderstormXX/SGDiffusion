# EXP7: Hessian-Guided Diffusion Control

Goal: test a practical use of the SGD diffusion theory. If Hessian-flat
directions mainly contribute neutral parameter-space wandering, then projecting
minibatch SGD updates away from these directions should reduce trajectory
dispersion without substantially changing train/test loss.

Compared methods:

- `baseline_sgd`: ordinary minibatch SGD with replacement.
- `suppress_flat`: remove the component of each minibatch gradient in the
  Hessian-flat eigenspace.
- `suppress_theory_high_variance`: remove the component in eigendirections with
  largest variance predicted from Hessian eigenvalues and minibatch gradient
  noise.
- `suppress_sharp`: control intervention removing the component in sharp
  Hessian eigendirections.

Primary metrics:

- final variance in flat and sharp eigenspaces;
- train loss, test loss, and generalization gap;
- reduction in flat variance relative to baseline;
- change in test loss relative to baseline.

This is an MLP-386 intervention experiment, not a large-model experiment.
