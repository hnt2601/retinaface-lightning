from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict as Adict
from iglovikov_helper_functions.metrics.map import recall_precision
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torchvision.ops import nms

from utils.box_utils import decode


class RetinaFaceModule(LightningModule):
    """Example of a `LightningModule` for face detection.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, config: Adict) -> None:
        """Initialize a `RetinaFaceModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # configure
        self.config = config

        # model
        self.prior_box = object_from_dict(
            self.config.prior_box, image_size=self.config.image_size
        )

        self.model = object_from_dict(self.config.model)

        # loss function
        self.loss_weights = self.config.loss_weights
        self.criterion = object_from_dict(self.config.loss, priors=self.prior_box)

    def setup(self, stage=0) -> None:
        """This hook is called on every process when using DDP."""
        pass

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of decode.
        """

        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        images = batch["image"]
        targets = batch["annotation"]

        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        predictions = self.forward(images)

        loss_localization, loss_classification, loss_landmarks = self.criterion(
            predictions, targets
        )

        total_loss = (
            self.loss_weights["localization"] * loss_localization
            + self.loss_weights["classification"] * loss_classification
            + self.loss_weights["landmarks"] * loss_landmarks
        )

        self.log(
            "train_classification",
            loss_classification,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_localization",
            loss_localization,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_landmarks",
            loss_landmarks,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            self._get_current_lr(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        data_dict = self._common_step(batch, batch_idx)

        image_height = data_dict["images"].shape[2]
        image_width = data_dict["images"].shape[3]
        annotations = data_dict["annotations"]
        file_names = data_dict["file_names"]
        predictions = data_dict["predictions"]

        gt_coco: List[Dict[str, Any]] = []

        for batch_id, annotation_list in enumerate(annotations):
            for annotation in annotation_list:
                x_min, y_min, x_max, y_max = annotation[:4]
                file_name = file_names[batch_id]

                gt_coco += [
                    {
                        "id": str(hash(f"{file_name}_{batch_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [
                            x_min.item() * image_width,
                            y_min.item() * image_height,
                            (x_max - x_min).item() * image_width,
                            (y_max - y_min).item() * image_height,
                        ],
                    }
                ]

        return OrderedDict({"predictions": predictions, "gt": gt_coco})

    def validation_epoch_end(self, outputs: List) -> None:
        result_predictions: List[dict] = []
        result_gt: List[dict] = []

        for output in outputs:
            result_predictions += output["predictions"]
            result_gt += output["gt"]

        _, _, average_precision = recall_precision(result_gt, result_predictions, 0.5)

        self.log(
            "epoch",
            self.trainer.current_epoch,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_loss", average_precision, on_step=False, on_epoch=True, logger=True
        )

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.from_numpy(np.array([lr]))[0].to(self.device)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single predict step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.model.eval()

        return self._common_step(batch, batch_idx)

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images = batch["image"].cuda()

        image_height = images.shape[2]
        image_width = images.shape[3]

        annotations = batch["annotation"]
        file_names = batch["file_name"]

        location, confidence, _ = self.forward(images)

        confidence = F.softmax(confidence, dim=-1)
        batch_size = location.shape[0]

        predictions_coco: List[Dict[str, Any]] = []

        scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(
            location.device
        )

        for batch_id in range(batch_size):
            boxes = decode(
                location.data[batch_id],
                self.prior_box.to(images.device),
                self.config.test_parameters.variance,
            )
            scores = confidence[batch_id][:, 1]

            valid_index = torch.where(scores > 0.1)[0]
            boxes = boxes[valid_index]
            scores = scores[valid_index]

            boxes *= scale

            # do NMS
            keep = nms(boxes, scores, self.config.val_parameters.iou_threshold)
            boxes = boxes[keep, :].cpu().numpy()

            if boxes.shape[0] == 0:
                continue

            scores = scores[keep].cpu().numpy()

            file_name = file_names[batch_id]

            for box_id, bbox in enumerate(boxes):
                x_min, y_min, x_max, y_max = bbox

                x_min = np.clip(x_min, 0, x_max - 1)
                y_min = np.clip(y_min, 0, y_max - 1)

                predictions_coco += [
                    {
                        "id": str(hash(f"{file_name}_{box_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": scores[box_id],
                    }
                ]

        return OrderedDict(
            {
                "predictions": predictions_coco,
                "images": images,
                "file_names": file_names,
                "annotations": annotations,
            }
        )

    def configure_optimizers(
        self,
    ) -> Tuple[
        Callable[[bool], Union[Optimizer, List[Optimizer], List[LightningOptimizer]]],
        List[Any],
    ]:
        optimizer = object_from_dict(
            self.config.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.config.scheduler, optimizer=optimizer)

        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]


if __name__ == "__main__":
    _ = RetinaFaceModule(None, None, None)
