#!/usr/bin/env bash

source $(pwd)/scripts/run_experiments.sh

DATA_PATH=data/resized128_image_fixed
DATA_LABEL_PATH=data/label_60c.csv

NAME=biggan_c60
BATCH_SIZE=32
WORKERS=8

ITERATION=10000
N_CLASSES=60
ACCUMULATIVE_STEPS=5

FEAT_G=24
FEAT_D=24

ADAM_BETA1=0
ADAM_BETA2=0.999

LOGGING_STEPS=100
CHECKPOINT_STEPS=1000

# running scripts
run_experiment