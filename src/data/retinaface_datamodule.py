from typing import Any, Dict, Optional

from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from data.data_augment import Preproc
from data.dataset import RetinaFaceDataset, detection_collate
from models.common import *


class RetinaFaceDatasetDataModule(LightningDataModule):
    """`LightningDataModule` for the RetinaFaceDataset dataset.

    The RetinaFaceDataset database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, config: Adict) -> None:
        """Initialize a `RetinaFaceDatasetDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # configure
        self.config = config

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of RetinaFaceDataset classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # RetinaFaceDataset(self.hparams.data_dir, train=True, download=True)
        # RetinaFaceDataset(self.hparams.data_dir, train=False, download=True)

        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        self.preproc = Preproc(img_dim=self.config.image_size[0])

        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            RetinaFaceDataset(
                label_path=TRAIN_LABEL_PATH,
                image_path=TRAIN_IMAGE_PATH,
                transform=from_dict(self.config.train_aug),
                preproc=self.preproc,
                rotate90=self.config.train_parameters.rotate90,
            ),
            batch_size=self.config.train_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            RetinaFaceDataset(
                label_path=VAL_LABEL_PATH,
                image_path=VAL_IMAGE_PATH,
                transform=from_dict(self.config.val_aug),
                preproc=self.preproc,
                rotate90=self.config.val_parameters.rotate90,
            ),
            batch_size=self.config.val_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        # return DataLoader(
        #     RetinaFaceDataset(
        #         label_path=TEST_LABEL_PATH,
        #         image_path=TEST_IMAGE_PATH,
        #         transform=from_dict(self.config.val_aug),
        #         preproc=self.preproc,
        #         rotate90=self.config.val_parameters.rotate90,
        #     ),
        #     batch_size=self.config.val_parameters.batch_size,
        #     num_workers=self.config.num_workers,
        #     shuffle=False,
        #     pin_memory=True,
        #     drop_last=True,
        #     collate_fn=detection_collate,
        # )

        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = RetinaFaceDatasetDataModule()
