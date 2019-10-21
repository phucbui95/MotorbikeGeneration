#!/usr/bin/env bash

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`

ROOT_DIR=$(pwd)
DATA_DIR=$ROOT_DIR/data
PREPROCESS_DIR=$DATA_DIR/resized128_image

KAGGLE_USERNAME=phucbb
KAGGLE_KEY=4f135a32eeaa77e0c425e6d54e8ba4f9