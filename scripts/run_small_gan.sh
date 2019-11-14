#!/usr/bin/env bash

source $(pwd)/scripts/min_gan_config.sh

export PYTHONPATH=$PYTHONPATH:./python

python python/gan_trainer.py \
    --name=$NAME \
    --path=$DATA_PATH \
    --label_path=$DATA_LABEL_PATH \
    --batch_size=$BATCH_SIZE \
    --workers=$WORKERS \
    --iteration=$ITERATION \
    --n_classes=$N_CLASSES \
    --accumulative_steps=$ACCUMULATIVE_STEPS \
    --feat_G=$FEAT_G \
    --feat_D=$FEAT_D \
    --beta1=$ADAM_BETA1 \
    --beta2=$ADAM_BETA2 \
    --logging_steps=$LOGGING_STEPS \
    --checkpoint_steps=$CHECKPOINT_STEPS



