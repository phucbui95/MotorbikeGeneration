#!/usr/bin/env bash

echo "==========================================="
echo "Prepare data"

install_tools() {
    apt-get install -y wget
    pip install awscli
    pip install tensorflow-gpu
}
download_source() {
    git init
    git remote add origin https://github.com/phucbui95/MotorbikeGeneration
    git add -A
    git config --global user.email "you@example.com"
    git commit -m "Backup"

    git checkout -b working
    git fetch --all
    git branch --set-upstream-to=origin/master working
    git reset --hard origin/master
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

echo "Installing tools"
install_tools
echo "Fetching source code from resposities"
download_source
echo "Downloading files from kaggle dataset"
download_preprocess
echo "Downloading data"
sh download.sh

# allow scripts
echo "Starting jupyter notebook"
cd /app || exit
nohup sh start_nb.sh &
nohup tensorboard --logdir=tensorboard_log &





