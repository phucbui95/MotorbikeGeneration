#!/usr/bin/env bash

source $(pwd)/scripts/run_experiments.sh
source $(pwd)/scripts/ds_config.sh

NAME=biggan_wdropout
BATCH_SIZE=32
IMAGE_SIZE=128
WORKERS=8

ITERATION=15000
N_CLASSES=30
ACCUMULATIVE_STEPS=8

FEAT_G=24
FEAT_D=24
LR_D=0.0008
LR_G=0.0004
USE_DROPOUT=0.2
CHECKPOINT_MODE=s3
# CKPT=checkpoint/biggan_c602019-11-15_11-13-06/model_0.pth

ADAM_BETA1=0
ADAM_BETA2=0.999

LOGGING_STEPS=200
CHECKPOINT_STEPS=1000
# export AWS_SHARED_CREDENTIALS_FILE=keys/credentials
# running scripts

# running scripts
if [[ -n $EVAL ]];
then
    echo "Running eval mode"
else
    run_experiment
fi