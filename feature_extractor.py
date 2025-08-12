import torch
import torch.nn as nn
import torchvision


class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Identity()
        self.features = torchvision.models.feature_extraction.create_feature_extractor(
            self.model,
            return_nodes={f"layer{k}": str(v) for v, k in enumerate([1, 2, 3, 4])},
        )

    def forward(self, x):
        features = self.features(x)
        return list(features.values())
