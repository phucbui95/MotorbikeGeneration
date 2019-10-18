DATA_DIR='./data'
ROOT_DIR='.'

echo "Creating data folder..."

mkdir -p $DATA_DIR

DATASET_FILE=$DATA_DIR/training_dataset.zip
if test -f "$DATASET_FILE"; then
    echo "$DATASET_FILE exist."
else
	wget -O $DATASET_FILE https://dl.challenge.zalo.ai/ZAC2019_GAN/training_dataset.zip
fi

EVALUATION_SCRIPT=$DATA_DIR/evaluation_script.zip
if test -f "$EVALUATION_SCRIPT"; then
    echo "$EVALUATION_SCRIPT exist."
else
	wget -O $EVALUATION_SCRIPT https://dl.challenge.zalo.ai/ZAC2019_GAN/evaluation_script.zip
fi

echo "Download completed."
echo ""

cd $DATA_DIR
unzip -o training_dataset.zip > /dev/null
unzip -o evaluation_script.zip > /dev/null

echo "Coping script for evaluation_script"
cd ..
cp -r $DATA_DIR/evaluation_script/client $ROOT_DIR
