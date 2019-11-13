#!/usr/bin/env bash

DATA_PATH=data/resized128_image/resized128_image_fixed/resized128_image_fixed
DATA_LABEL_PATH=data/resized128_image/resized128_image_fixed/label.csv

BATCH_SIZE=2
WORKERS=1

ITERATION=10
N_CLASSES=30
ACCUMULATIVE_STEPS=1
FEAT_G=24
FEAT_D=24

ADAM_BETA1=0
ADAM_BETA2=0.999