#!/usr/bin/env bash

source $(pwd)/scripts/run_experiments.sh
source $(pwd)/scripts/ds_config.sh

NAME=fast_and_furious_v1
BATCH_SIZE=32
IMAGE_SIZE=128
WORKERS=8

ITERATION=30000
N_CLASSES=30
ACCUMULATIVE_STEPS=5

LOSS=hinge
FEAT_G=42
FEAT_D=42
USE_ATTENTION=0
CODE_DIM=64
LATENT_SIZE=512

LR_D=0.0004
LR_G=0.0004
# USE_DROPOUT=
CHECKPOINT_MODE=s3
USE_DROPOUT=-1
# CKPT=checkpoint/biggan_vallia2019-11-19_16-19-04/model_5000.pth

ADAM_BETA1=0
ADAM_BETA2=0.9

LOGGING_STEPS=100
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
