#!/usr/bin/env bash

run_experiment() {
    export PYTHONPATH=$PYTHONPATH:./python

    python python/gan_trainer.py \
        --name=$NAME \
        --path=$DATA_PATH \
        --label_path=$DATA_LABEL_PATH \
        --batch_size=$BATCH_SIZE \
        --workers=$WORKERS \
        --shuffle \
        --iteration=$ITERATION \
        --n_classes=$N_CLASSES \
        --accumulative_steps=$ACCUMULATIVE_STEPS \
        --image_size=$IMAGE_SIZE \
        --loss=$LOSS \
        --feat_G=$FEAT_G \
        --feat_D=$FEAT_D \
        --beta1=$ADAM_BETA1 \
        --beta2=$ADAM_BETA2 \
        --lr_D=$LR_D \
        --lr_G=$LR_G \
        --use_dropout=$USE_DROPOUT \
        --logging_steps=$LOGGING_STEPS \
        --checkpoint_steps=$CHECKPOINT_STEPS \
        --checkpoint_mode=$CHECKPOINT_MODE \
        --ckpt=$CKPT
}