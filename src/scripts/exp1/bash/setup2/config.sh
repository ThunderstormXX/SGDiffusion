#!/bin/bash
export DATASET_TRAIN="mnist_similar"
export DATASET_VAL="mnist"
export MODEL="mlp"
export BATCH_SIZE=64
export LR=0.1
export EPOCHS_SGD=1000
export EPOCHS_GD=1000
export SEED=42
export HESSIAN_STEPS=1000
export MANY_RUNS_STEPS=1111
export MANY_RUNS_SAMPLES=1000
export LRS_LIST="0.1,0.01"