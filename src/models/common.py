from pathlib import Path
import os


TRAIN_IMAGE_PATH = Path(os.getenv("TRAIN_IMAGE_PATH", default=""))
VAL_IMAGE_PATH = Path(os.getenv("VAL_IMAGE_PATH", default=""))

TRAIN_LABEL_PATH = Path(os.getenv("TRAIN_LABEL_PATH", default=""))
VAL_LABEL_PATH = Path(os.getenv("VAL_LABEL_PATH", default=""))
