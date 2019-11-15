#!/usr/bin/env bash

#!/usr/bin/env bash

source $(pwd)/scripts/run_experiments.sh

ROOT_DATA=data
DATA_PATH=$ROOT_DATA/resized128_image_fixed
DATA_LABEL_PATH=$ROOT_DATA/label.csv

NAME=biggan_c60
BATCH_SIZE=4
IMAGE_SIZE=64
WORKERS=8

ITERATION=10
N_CLASSES=30
ACCUMULATIVE_STEPS=1

FEAT_G=24
FEAT_D=24
CKPT=checkpoint/biggan_c602019-11-15_11-13-06/model_0.pth

ADAM_BETA1=0
ADAM_BETA2=0.999

LOGGING_STEPS=100
CHECKPOINT_STEPS=1000

# running scripts
run_experiment