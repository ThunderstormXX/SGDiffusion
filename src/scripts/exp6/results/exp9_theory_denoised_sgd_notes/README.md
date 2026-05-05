# EXP9 Rationale: Theory-Denoised SGD

EXP8 showed that amplifying theory-selected diffusion changes variance but did
not improve convergence in the MLP-386 full-lite setting. EXP7 showed that
removing the whole update in those directions reduces diffusion but slightly
hurts optimization, which confounds deterministic drift and stochastic noise.

EXP9 isolates the stochastic part. It uses the full-batch gradient as a control
variate and removes only minibatch noise in directions where the theory predicts
large SGD variance.

The intended practical claim is not "always remove SGD noise", but:

> The theory identifies where SGD stochasticity lives, and this can be used to
> build directional control-variate optimizers.

Positive result criteria:

- large reduction in theory-selected variance;
- no worse, or better, train/test loss than baseline;
- better behavior than random-direction or sharp-direction denoising controls.

Negative result criteria:

- denoising theory-selected directions worsens loss;
- random denoising performs similarly;
- variance is reduced but optimization behavior is unchanged.
