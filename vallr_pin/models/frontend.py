"""视觉前端：3D 卷积 stem + 2D ResNet 主干。

输入是唇部 ROI 灰度视频 (B, T, 1, H, W)，H=W=88 为 VSR 领域惯例。
3D stem 只在空间维下采样、**时间维保持 1:1**，因为汉字发音速率 (~4-6 字/秒) 与
视频帧率 (25fps) 之比只有 4~6，过度时间下采样会让 CTC 的 T' >= L 约束失效。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, down=None):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.relu, self.down = nn.ReLU(inplace=True), down

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, cout, stride=1, down=None):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = conv3x3(cout, cout, stride), nn.BatchNorm2d(cout)
        self.conv3 = nn.Conv2d(cout, cout * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout * 4)
        self.relu, self.down = nn.ReLU(inplace=True), down

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


class ResNetTrunk(nn.Module):
    def __init__(self, block, layers: List[int], width: int = 64, in_ch: int = 64):
        super().__init__()
        self.inplanes = in_ch
        self.layer1 = self._make(block, width, layers[0], 1)
        self.layer2 = self._make(block, width * 2, layers[1], 2)
        self.layer3 = self._make(block, width * 4, layers[2], 2)
        self.layer4 = self._make(block, width * 8, layers[3], 2)
        self.out_dim = width * 8 * block.expansion

    def _make(self, block, planes, blocks, stride):
        down = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, down)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


_ARCH = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}


class VisualFrontend(nn.Module):
    """(B,T,1,H,W) -> (B,T,d_model)"""

    def __init__(self, arch: str = "resnet18", d_model: int = 256, in_channels: int = 1,
                 width: int = 64, dropout: float = 0.0):
        super().__init__()
        if arch not in _ARCH:
            raise ValueError(f"unknown frontend arch: {arch}")
        block, layers = _ARCH[arch]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.trunk = ResNetTrunk(block, layers, width=width, in_ch=64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.trunk.out_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        b, t = video.shape[:2]
        x = video.permute(0, 2, 1, 3, 4)            # (B,C,T,H,W)
        x = self.stem(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, x.size(1), x.size(3), x.size(4))
        x = self.trunk(x)
        x = self.pool(x).flatten(1)                 # (B*T, C)
        x = self.proj(self.drop(x))
        return x.view(b, t, self.d_model)
