#!/usr/bin/env bash

APP_PATH=/app

SCRIPT_PATH=$APP_PATH/scripts/generate_images.sh
CONFIG_FILE=$APP_PATH/scripts/best_conf.sh
MODEL=$APP_PATH/final_model/best.pth

cd $APP_PATH || echo "APP_PATH not found"
"$SCRIPT_PATH" $CONFIG_FILE $MODEL

echo "Copying file to output directory"
mv $APP_PATH/outputs/output_images /result/
