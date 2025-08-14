import torch
import torch.nn as nn
import torchvision
import torchvision.models.feature_extraction


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


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone
        self.model = torchvision.models.vit_b_16(weights="DEFAULT")
        self.model.heads = nn.Identity()
        self.features = torchvision.models.feature_extraction.create_feature_extractor(
            self.model,
            return_nodes={
                f"encoder.layers.encoder_layer_{i}": str(i) for i in range(2, 12, 3)
            },
        )

    def forward(self, x):
        features = self.features(x)
        return [f[:, 1:] for f in features.values()]


class HybridFeatureExtractor(nn.Module):
    def __init__(self, cnn_backbone=None, transformer_backbone=None):
        super().__init__()

        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone

        self.cnn_extractor = CNNFeatureExtractor(cnn_backbone)
        self.transformer_extractor = TransformerFeatureExtractor(transformer_backbone)

    def forward(self, x):
        cnn_features = self.cnn_extractor(x)[:2]
        transformer_features = self.transformer_extractor(x)[2:]
        return cnn_features + transformer_features


if __name__ == "__main__":
    extractor = CNNFeatureExtractor().cuda()
    inp = torch.randn(1, 3, 256, 256).cuda()
    outs = extractor(inp)
    for i, out in outs:
        print(f"{i}: {out.shape}")
