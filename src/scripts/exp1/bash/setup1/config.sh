#!/bin/bash
export DATASET_TRAIN="mnist"
export DATASET_VAL="mnist"
export MODEL="mlp"
export BATCH_SIZE=64
export LR=0.1
export EPOCHS_SGD=10
export EPOCHS_GD=10
export SEED=42
export HESSIAN_STEPS=10
export MANY_RUNS_STEPS=11
export MANY_RUNS_SAMPLES=100
export LRS_LIST="0.1,0.01"