import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            out += x
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
            align_corners=False,True
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
