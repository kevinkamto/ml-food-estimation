import timm
import torch
import torch.nn as nn


class DualStreamEfficientNet(nn.Module):
    def __init__(self, num_classes=34, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features  # 1280

        self.fusion = nn.Sequential(nn.Linear(feature_dim * 2, 1024), nn.ReLU(), nn.Dropout(0.3))

        self.regression_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, before, after):
        feat_before = self.backbone(before)
        feat_after = self.backbone(after)
        fused = self.fusion(torch.cat([feat_before, feat_after], dim=1))
        leftover_norm = self.regression_head(fused).squeeze(1)
        category_logits = self.classification_head(fused)
        return leftover_norm, category_logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
