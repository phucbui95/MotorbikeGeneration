#!/usr/bin/env bash

source $(pwd)/scripts/min_gan_config.sh

export PYTHONPATH=$PYTHONPATH:./python

python python/gan_trainer.py \
    --name=testing \
    --path=$DATA_PATH \
    --label_path=$DATA_LABEL_PATH \
    --batch_size=2 \
    --workers=$WORKERS \
    --iteration=$ITERATION \
    --n_classes=$N_CLASSES \
    --accumulative_steps=$ACCUMULATIVE_STEPS \
    --feat_G=$FEAT_G \
    --feat_D=$FEAT_D \
    --beta1=$ADAM_BETA1 \
    --beta1=$ADAM_BETA2



