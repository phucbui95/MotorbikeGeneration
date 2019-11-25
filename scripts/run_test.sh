#!/usr/bin/env bash

source $(pwd)/scripts/run_experiments.sh
source $(pwd)/scripts/ds_config.sh

NAME=biggan_c60
BATCH_SIZE=4
IMAGE_SIZE=64
WORKERS=8

LOSS=crhinge
ITERATION=100
N_CLASSES=30
ACCUMULATIVE_STEPS=1
USE_ATTENTION=0

LATENT_SIZE=256
LOSS=hinge
FEAT_G=24
FEAT_D=24
CODE_DIM=32

LR_D=0.0002
LR_G=0.0002

USE_DROPOUT=0.2
CHECKPOINT_MODE=local
CKPT=checkpoint/biggan_c602019-11-15_11-13-06/model_0.pth

ADAM_BETA1=0
ADAM_BETA2=0.999

LOGGING_STEPS=10
CHECKPOINT_STEPS=1000
export AWS_SHARED_CREDENTIALS_FILE=keys/credentials
# running scripts

# running scripts
if [[ -n $EVAL ]];
then
    echo "Running eval mode"
else
    run_experiment
fi