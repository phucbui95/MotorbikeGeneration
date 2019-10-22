#!/usr/bin/env bash
ROOT_DIR=$(pwd)
DATA_DIR=$ROOT_DIR/data
PREPROCESSED_DIR=$DATA_DIR/resized128_image

echo "Configure ROOT_DIR: $ROOT_DIR"
echo "Configure DATA_DIR: $DATA_DIR"

echo "Creating data folder..."

mkdir -p $DATA_DIR

init_training_dataset() {
    DATASET_FILE=$DATA_DIR/training_dataset.zip
    if test -f "$DATASET_FILE"; then
        echo "$DATASET_FILE exist."
    else
        wget -O $DATASET_FILE https://dl.challenge.zalo.ai/ZAC2019_GAN/training_dataset.zip
    fi

    echo "Download completed."
    echo ""
    cd $DATA_DIR
    unzip -o training_dataset.zip > /dev/null
}

init_evaluation_scripts() {
    EVALUATION_SCRIPT=$DATA_DIR/evaluation_script.zip
    if test -f "$EVALUATION_SCRIPT"; then
        echo "$EVALUATION_SCRIPT exist."
    else
        wget -O $EVALUATION_SCRIPT https://dl.challenge.zalo.ai/ZAC2019_GAN/evaluation_script.zip
    fi

    cd $DATA_DIR
    unzip -o evaluation_script.zip > /dev/null
    echo "Coping script for evaluation_script"
    cd ..
    cp -r $DATA_DIR/evaluation_script/client $ROOT_DIR
}
echo "Result $PREPROCESSED_DIR"

if [ -d $PREPROCESSED_DIR ]; then
    echo "Dataset already existed"
else
    init_training_dataset
    echo "Preprocessing data"
    python run_notebook Dataset.ipynb
fi

init_evaluation_scripts