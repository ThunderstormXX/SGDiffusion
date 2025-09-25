Отлично 👌 Дополнил README с указанием, что запускать можно через `src/scripts/exp2/bash/run_full_setup.sh`. Вот финальная версия:

# Neural Network Optimization Experiments

## Description

We evaluate our theoretical framework through controlled experiments in computer vision and natural language processing. Models are first trained with SGD and then refined with GD to reach local minima. Along SGD trajectories, we compute Hessians and project dynamics into the eigenbasis to measure dispersion across curvature directions. These experiments reveal (i) variance saturation in sharp directions (inverse Einstein relation), (ii) indefinite diffusion in flat directions, and (iii) robustness of these effects across architectures and datasets.

The **main experiments** are located in:

* `src/scripts/exp1/`
* `src/scripts/exp2/`

---

## Installation and Setup

### 1. Dependencies

Make sure you have Python 3.9+ installed, then run:

* `pip install uv`
* `uv venv`
* `uv sync`
* `source .venv/bin/activate`

### 2. Running Experiments

You can run experiments with:

* `bash src/scripts/exp{i}/bash/run_full_setup.sh`
