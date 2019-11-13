#!/usr/bin/env bash

DATA_PATH=data/resized128_image_fixed
DATA_LABEL_PATH=data/label.csv


export PYTHONPATH=$PYTHONPATH:./python

python python/gan_trainer.py \
    --name=testing \
    --path=$DATA_PATH \
    --label_path=$DATA_LABEL_PATH \
    --batch_size=2 \
    --iteration=10 \

