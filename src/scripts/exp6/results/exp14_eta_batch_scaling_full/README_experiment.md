# EXP14: Eta and batch-size scaling

MLP-386 learning-rate and batch-size scaling of projected variance in flat and sharp Hessian buckets.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp14_eta_batch_scaling_full/config.yaml
```

## Artifacts

- `config.yaml`: exact configuration used for this run.
- `environment.json`: Python, package, hardware, git metadata.
- `runtime.json`: runtime and RSS memory snapshot.
- `metrics.json`: machine-readable primary metrics.
- `raw_outputs.npz`: raw trajectories/statistics.
- `figure_data.csv`: plotted data with mean/std/95% CI when applicable.
- `make_figure.py`: figure generation from saved artifacts only.

## Primary Metrics

```json
{
  "batch_grid": [
    32,
    64,
    128
  ],
  "flat_slope_inv_batch_correlation": 0.6810163730752108,
  "flat_slope_lr_correlation": 0.9552988495925211,
  "lr_0.0125_batch_128_flat_late_mean_variance": 2.274720559483634e-12,
  "lr_0.0125_batch_128_flat_late_slope": 2.0091154140292642e-14,
  "lr_0.0125_batch_128_sharp_late_mean_variance": 1.9108341803075746e-05,
  "lr_0.0125_batch_128_sharp_late_slope": 4.2404820175891405e-08,
  "lr_0.0125_batch_32_flat_late_mean_variance": 7.119770051300378e-12,
  "lr_0.0125_batch_32_flat_late_slope": 4.3783252693028807e-14,
  "lr_0.0125_batch_32_sharp_late_mean_variance": 7.988269499037415e-05,
  "lr_0.0125_batch_32_sharp_late_slope": 3.5532048059394594e-07,
  "lr_0.0125_batch_64_flat_late_mean_variance": 2.8746293539994028e-12,
  "lr_0.0125_batch_64_flat_late_slope": 2.1635334892681037e-14,
  "lr_0.0125_batch_64_sharp_late_mean_variance": 5.134337698109448e-05,
  "lr_0.0125_batch_64_sharp_late_slope": 1.7165812096209266e-07,
  "lr_0.025_batch_128_flat_late_mean_variance": 9.520323765443983e-12,
  "lr_0.025_batch_128_flat_late_slope": 8.697799734911202e-14,
  "lr_0.025_batch_128_sharp_late_mean_variance": 5.6515233154641464e-05,
  "lr_0.025_batch_128_sharp_late_slope": 7.467200962959689e-08,
  "lr_0.025_batch_32_flat_late_mean_variance": 2.971943177665359e-11,
  "lr_0.025_batch_32_flat_late_slope": 1.8157525334485021e-13,
  "lr_0.025_batch_32_sharp_late_mean_variance": 0.0002478826208971441,
  "lr_0.025_batch_32_sharp_late_slope": 7.648135838930373e-07,
  "lr_0.025_batch_64_flat_late_mean_variance": 1.2315979833199542e-11,
  "lr_0.025_batch_64_flat_late_slope": 1.0275795901186284e-13,
  "lr_0.025_batch_64_sharp_late_mean_variance": 0.00015787396114319563,
  "lr_0.025_batch_64_sharp_late_slope": 1.5653848614809845e-07,
  "lr_0.05_batch_128_flat_late_mean_variance": 4.3220076823002884e-11,
  "lr_0.05_batch_128_flat_late_slope": 4.334069747436677e-13,
  "lr_0.05_batch_128_sharp_late_mean_variance": 0.0001365023636026308,
  "lr_0.05_batch_128_sharp_late_slope": 1.5695950423833093e-07,
  "lr_0.05_batch_32_flat_late_mean_variance": 1.318817000051098e-10,
  "lr_0.05_batch_32_flat_late_slope": 8.32911715788396e-13,
  "lr_0.05_batch_32_sharp_late_mean_variance": 0.00063987827161327,
  "lr_0.05_batch_32_sharp_late_slope": 5.792622687295119e-07,
  "lr_0.05_batch_64_flat_late_mean_variance": 5.692121080036472e-11,
  "lr_0.05_batch_64_flat_late_slope": 5.964631938161683e-13,
  "lr_0.05_batch_64_sharp_late_mean_variance": 0.0003947542863897979,
  "lr_0.05_batch_64_sharp_late_slope": -1.3156392794501527e-06,
  "lr_0.1_batch_128_flat_late_mean_variance": 2.5516730395303e-10,
  "lr_0.1_batch_128_flat_late_slope": 2.98292258948951e-12,
  "lr_0.1_batch_128_sharp_late_mean_variance": 0.00025614671176299453,
  "lr_0.1_batch_128_sharp_late_slope": 4.987918925637933e-07,
  "lr_0.1_batch_32_flat_late_mean_variance": 7.133353352450911e-10,
  "lr_0.1_batch_32_flat_late_slope": 5.163489873767037e-12,
  "lr_0.1_batch_32_sharp_late_mean_variance": 0.0012384022120386362,
  "lr_0.1_batch_32_sharp_late_slope": -4.363510691161649e-06,
  "lr_0.1_batch_64_flat_late_mean_variance": 3.6054875751645454e-10,
  "lr_0.1_batch_64_flat_late_slope": 5.861004241460213e-12,
  "lr_0.1_batch_64_sharp_late_mean_variance": 0.0006893352838233113,
  "lr_0.1_batch_64_sharp_late_slope": -7.63687026587181e-06,
  "lr_grid": [
    0.0125,
    0.025,
    0.05,
    0.1
  ],
  "n_runs": 12,
  "pass": true,
  "steps": 200
}
```