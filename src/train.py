import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from addict import Dict as Adict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning.loggers import WandbLogger

from models.retinaface_module import RetinaFaceModule
from data.retinaface_datamodule import RetinaFaceDatasetDataModule


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    with args.config_path.open() as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    pl.trainer.seed_everything(config.seed, workers=True)

    model = RetinaFaceModule(config)
    datamodule = RetinaFaceDatasetDataModule(config)

    Path(config.checkpoint_callback.dirpath).mkdir(exist_ok=True, parents=True)

    # TODO: add callback early stopping

    trainer = object_from_dict(
        config.trainer,
        logger=WandbLogger(config.experiment_name),
        callbacks=[object_from_dict(config.checkpoint_callback)],
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
