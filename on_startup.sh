#!/usr/bin/env bash



echo "==========================================="
echo "Prepare data"

download_source() {
    git init
    git remote add origin https://github.com/phucbui95/MotorbikeGeneration
    git pull
}

source configure.sh
echo "Configure ROOT_DIR: $(pwd)"

download_preprocess() {
    mkdir -p $DATA_DIR
    cd $DATA_DIR
    kaggle datasets download -d phucbb/preprocesmotorbike
    unzip -o -d resized128_image preprocesmotorbike.zip > /dev/null
    cd $ROOT_DIR
}

download_preprocess

# Preprocessing data
if test -f "$PREPROCESSED_DIR"; then
    # download and unzip data
    sh download.sh
    echo "Preprocessing data"
    python run_notebook Dataset.ipynb
else
    echo "Preprocessed data have already downloaded"
fi






