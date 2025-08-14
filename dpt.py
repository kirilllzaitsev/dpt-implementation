import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import CNNFeatureExtractor
from utils.misc import to_tuple


class ResBlock(nn.Module):
    def __init__(self, inch, outch):
        super(ResBlock, self).__init__()
        self.inch = inch
        self.outch = outch
        self.conv1 = nn.Conv2d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(outch, outch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outch)
        self.bn2 = nn.BatchNorm2d(outch)
        if inch != outch:
            self.shortcut = nn.Conv2d(inch, outch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # conv -> bn -> act -> dropout
        assert x.shape[1] == self.inch
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        if self.shortcut is not None:
            x = self.shortcut(x)
            out = out + x
        return out


class ReassembleBlock(nn.Module):
    def __init__(self, s, inch, outch, hw):
        super(ReassembleBlock, self).__init__()
        self.s = s
        self.hw = hw
        self.inch = inch
        self.outch = outch
        # mimics convolutional feature extractor when it is actually a transformer
        self.resample = functools.partial(
            F.interpolate,
            size=(hw[0] // s, hw[1] // s),
            mode="bilinear",
            align_corners=True,
        )
        self.proj = nn.Conv2d(inch, outch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, n, d = x.shape
        p = int(math.sqrt(n))
        assert p * p == n, (p, n)
        x = x.reshape(b, p, p, d).permute(0, 3, 1, 2)
        x = self.resample(x)
        x = self.proj(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, inch, outch):
        super(FusionBlock, self).__init__()
        self.inch = inch
        self.outch = outch
        self.downsample = functools.partial(
            F.interpolate, scale_factor=2, mode="bilinear", align_corners=True
        )
        self.rb1 = ResBlock(inch, inch)
        self.rb2 = ResBlock(inch, outch)

    def forward(self, x1, x2=None):
        # x1=left features, x2=top features. both with identical dimensions
        if x2 is None:
            x2 = torch.zeros_like(x1)
        assert x1.shape[1] == self.inch and x2.shape[1] == self.inch, (
            x1.shape,
            x2.shape,
        )
        assert x1.shape == x2.shape
        x = self.rb1(x1) + x2
        x = self.rb2(x)
        x = self.downsample(x)
        return x


class HeadBlock(nn.Module):
    def __init__(self, inch, outch, scale_factor=None):
        super(HeadBlock, self).__init__()
        self.inch = inch
        self.outch = outch
        self.scale_factor = scale_factor

        self.head = nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inch, outch, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.scale_factor is not None:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )

        x = self.head(x)
        return x


class DPT(nn.Module):
    def __init__(self, hw, extractor_name="resnet50"):
        super().__init__()

        self.hw = to_tuple(hw)

        if "vit_" in extractor_name:
            self.extractor = TransformerFeatureExtractor(backbone=extractor_name)
        elif extractor_name == "hybrid":
            self.extractor = HybridFeatureExtractor(
                cnn_backbone="resnet50", transformer_backbone="vit_b_16"
            )
        else:
            self.extractor = CNNFeatureExtractor(backbone=extractor_name)

        self.transformer_block4 = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=1024, dropout=0.0
        )
        self.transformer_block3 = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0
        )
        self.transformer_block2 = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=1024, dropout=0.0
        )
        self.transformer_block1 = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=1024, dropout=0.0
        )
        self.reassemble_block4 = ReassembleBlock(s=32, inch=512, outch=128, hw=self.hw)
        self.reassemble_block3 = ReassembleBlock(s=16, inch=256, outch=128, hw=self.hw)
        self.reassemble_block2 = ReassembleBlock(s=8, inch=128, outch=128, hw=self.hw)
        self.reassemble_block1 = ReassembleBlock(s=4, inch=64, outch=128, hw=self.hw)
        self.fusion_block4 = FusionBlock(inch=128, outch=128)
        self.fusion_block3 = FusionBlock(inch=128, outch=128)
        self.fusion_block2 = FusionBlock(inch=128, outch=128)
        self.fusion_block1 = FusionBlock(inch=128, outch=128)
        self.head_block = HeadBlock(inch=128, outch=1, scale_factor=2)

    def forward(self, x):
        features = self.extractor(x)
        tokens1, tokens2, tokens3, tokens4 = features
        top4 = self.fusion_block4(
            self.reassemble_block4(self.transformer_block4(tokens4))
        )
        top3 = self.fusion_block3(
            self.reassemble_block3(self.transformer_block3(tokens3)), top4
        )
        top2 = self.fusion_block2(
            self.reassemble_block2(self.transformer_block2(tokens2)), top3
        )
        top1 = self.fusion_block1(
            self.reassemble_block1(self.transformer_block1(tokens1)), top2
        )
        depth = self.head_block(top1)
        return {"depth": depth}


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    hw = x.shape[-2:]
    dpt = DPT(hw=hw)
    out = dpt(x)
    depth = out["depth"]
    print(f"{depth.shape=}")
    assert depth.shape == (1, 1, *hw)
