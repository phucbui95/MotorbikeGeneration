#!/usr/bin/env bash
EVAL=1
CONFIG_FILE=$1
source $CONFIG_FILE

CKPT=$2

echo "Generating images configure $CONFIG_FILE"
echo "Loading from checkpoint $CKPT"

export PYTHONPATH=$PYTHONPATH:./python

python python/generate_images.py \
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
        --ckpt=$CKPT \
        --use_attention=$USE_ATTENTION \
        --code_dim=$CODE_DIM \
        --latent_size=$LATENT_SIZE