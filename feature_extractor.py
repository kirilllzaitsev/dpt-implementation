import torch
import torch.nn as nn
import torchvision
import torchvision.models.feature_extraction


class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet50", weights="DEFAULT"):
        super().__init__()
        self.backbone = backbone

        if backbone == "resnet18":
            self.model = torchvision.models.resnet18(weights=weights)
        elif backbone == "resnet50":
            self.model = torchvision.models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.model.fc = nn.Identity()
        self.features = torchvision.models.feature_extraction.create_feature_extractor(
            self.model,
            return_nodes={f"layer{k}": str(v) for v, k in enumerate([1, 2, 3, 4])},
        )

    def forward(self, x):
        features = self.features(x)
        return [f.flatten(-2, -1).transpose(-2, -1) for f in features.values()]


class TransformerFeatureExtractor(nn.Module):

    def __init__(self, backbone="vitb", weights="DEFAULT", image_size=224):
        if weights == "DEFAULT":
            assert image_size == 224, "To use pretrained, image_size must be 224"

        super().__init__()
        self.backbone = backbone

        model_init_kwargs = {"weights": weights, "image_size": image_size}
        if backbone == "vitb":
            self.model = torchvision.models.vit_b_16(**model_init_kwargs)
        elif backbone == "vitl":
            self.model = torchvision.models.vit_l_16(**model_init_kwargs)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

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
    def __init__(
        self,
        cnn_backbone="resnet50",
        transformer_backbone="vitb",
        transformer_kwargs=None,
    ):
        super().__init__()

        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone
        self.transformer_kwargs = (
            {} if transformer_kwargs is None else transformer_kwargs
        )

        self.cnn_extractor = CNNFeatureExtractor(cnn_backbone)
        self.transformer_extractor = TransformerFeatureExtractor(
            transformer_backbone, **self.transformer_kwargs
        )

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
