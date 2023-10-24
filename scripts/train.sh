export TRAIN_IMAGE_PATH=./data/widerface/train/images
export VAL_IMAGE_PATH=./data/widerface/val/images
export TRAIN_LABEL_PATH=./data/widerface/train/label.json
export VAL_LABEL_PATH=./data/widerface/val/label.json

python src/train.py -c ./configs/config.yaml