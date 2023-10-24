import os
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision.models import _utils

from models.backbone import FPN, SSH, MobileNetV1
from models.head import BboxHead, ClassHead, LandmarkHead

pwd = os.getcwd()


class RetinaFace(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool,
        in_channels: int,
        return_layers: Dict[str, int],
        out_channels: int,
    ) -> None:
        super().__init__()

        backbone = MobileNetV1()
        if pretrained:
            checkpoint = torch.load(
                os.path.join(pwd, "weights/mobilenetV1X0.25_pretrain.tar"),
                map_location=torch.device("cpu"),
            )
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            backbone.load_state_dict(new_state_dict)

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = in_channels
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=out_channels)
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=3, in_channels=out_channels
        )

    @staticmethod
    def _make_class_head(
        fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
    ) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(
        fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
    ) -> nn.ModuleList:
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(
        fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
    ) -> nn.ModuleList:
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )

        return bbox_regressions, classifications, ldm_regressions
