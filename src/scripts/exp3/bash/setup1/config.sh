#!/bin/bash
# Dataset and model parameters
export DATASET_TRAIN="mnist"
export DATASET_VAL="mnist"
export MODEL="mlp"
export BATCH_SIZE=64
export SAMPLE_SIZE=6400
export SEED=4

# Model architecture parameters - empty to use base model from exp1
export MODEL_PARAMS='{}'
# Training parameters
export LR=0.01
export EPOCHS_SGD=1000
export EPOCHS_GD=1000
export HESSIAN_STEPS=10
export MANY_RUNS_STEPS=11
export MANY_RUNS_SAMPLES=10
export DATALOADER="default"
export GD_SCALING="default" #"N/B" # samples size / batch size 
export LRS_LIST="0.05,0.01,0.005,0.001" #"0.1,0.01,0.001,0.0001"
export DTYPE="float64"
export DEVICE='cpu'

# Optimizers configuration
# Format: "name1,name2,name3,..."
export OPTIMIZERS="sgd,adam,rmsprop,muon,signsgd"

# Optimizer hyperparameters as JSON
export OPTIMIZER_PARAMS='{"sgd":{"lr":0.01},"adam":{"lr":0.01,"betas":[0.9,0.999],"eps":1e-8},"rmsprop":{"lr":0.01,"alpha":0.99,"eps":1e-8},"muon":{"lr":0.01,"momentum":0.9,"dampening":0,"noise_scale":0.01,"noise_type":"gaussian"},"signsgd":{"lr":0.01}}'

# Valley exploration parameters
export VALLEY_STEPS=1000  # Number of steps for valley exploration with SGD