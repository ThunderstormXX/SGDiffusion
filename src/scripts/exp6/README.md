# EXP6 Reproducible Experiments

This directory contains a reproducible experimental pipeline for the paper
“Why SGD is not Brownian Motion”.

## Layout

- `configs/`: smoke and full YAML configs for numbered experiments.
- `src/`: shared Python implementation.
- `scripts/`: entrypoints and figure generation.
- `results/`: generated experiment artifacts.
- `figures/`: reserved for copied publication figures.
- `tables/`: generated machine-readable summary tables.

## One-command Reproduction

Smoke run:

```bash
bash src/scripts/exp6/reproduce_all.sh
```

Full run:

```bash
MODE=full bash src/scripts/exp6/reproduce_all.sh
```

EXP4 is intentionally disabled by default because the requested modern-model
Lanczos/HVP experiment is long:

```bash
MODE=full RUN_LONG=1 bash src/scripts/exp6/reproduce_all.sh
```

## Contract

Each completed experiment writes:

- `config.yaml`
- `environment.json`
- `runtime.json`
- `metrics.json`
- `raw_outputs.npz`
- `figure_data.csv`
- `make_figure.py`
- `figure.png`
- `README_experiment.md`

Figures are generated only from saved intermediate artifacts.

