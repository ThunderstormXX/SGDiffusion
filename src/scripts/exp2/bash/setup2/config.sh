#!/bin/bash
export DATASET_TRAIN="shakespeare"
export DATASET_VAL="shakespeare"
export MODEL="nanogpt"
export BATCH_SIZE=64
export LR=0.01
export EPOCHS_SGD=10000
export EPOCHS_GD=10000
export SEED=42
export HESSIAN_STEPS=1000
export MANY_RUNS_STEPS=1111
export MANY_RUNS_SAMPLES=1000
export DATALOADER="replacement"
export GD_SCALING="default"
export LRS_LIST="0.003,0.001,0.0005,0.0001"
# export LRS_LIST="0.01"
export DTYPE="float32"
export DEVICE='cpu