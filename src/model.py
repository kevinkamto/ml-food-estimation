from typing import cast

import timm
import torch
import torch.nn as nn


class DualStreamEfficientNet(nn.Module):
    """
    Single-task dual-stream EfficientNet-B0.

    Fusion concatenates:
      feat_before  (1280,)
      feat_after   (1280,)
      |difference| (1280,)
      area_ratio   (1,)    -- non-black pixel fraction: after_mask / before_mask

    Regression head outputs consumption ratio r = w_after / w_before in [0, 1].
    Denormalize at inference: w_after_hat = r_hat * w_before.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        feat_dim: int = cast(int, self.backbone.num_features)  # 1280

        # feat_before + feat_after + |diff| + area_ratio scalar
        fusion_in: int = feat_dim * 3 + 1

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(512, 1),
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor, area_ratio: torch.Tensor
    ) -> torch.Tensor:
        feat_before = self.backbone(before)
        feat_after = self.backbone(after)
        diff = torch.abs(feat_before - feat_after)

        if area_ratio.dim() == 1:
            area_ratio = area_ratio.unsqueeze(1)

        fused = self.fusion(torch.cat([feat_before, feat_after, diff, area_ratio], dim=1))
        out: torch.Tensor = self.regression_head(fused).squeeze(1).clamp(0.0, 1.0)
        return out

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
