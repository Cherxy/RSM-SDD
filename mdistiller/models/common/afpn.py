from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _basic_conv(filter_in: int, filter_out: int, kernel_size: int, stride: int = 1, pad: int | None = None):
    if pad is None:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels=filter_in,
                        out_channels=filter_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        bias=False,
                    ),
                ),
                ("bn", nn.BatchNorm2d(num_features=filter_out)),
                ("relu", nn.ReLU(inplace=True)),
            ]
        )
    )


class _BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.proj = _basic_conv(in_channels, out_channels, 1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True,
        )


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.downsample = _basic_conv(in_channels, out_channels, stride, stride, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class _ASF2(nn.Module):
    """
    Adaptive spatial fusion for two adjacent scales.

    The supplied partial code used softmax-based ASFF. The thesis explicitly
    defines a sigmoid-gated complementary fusion, so the engineering
    implementation follows the thesis while preserving the progressive fusion
    structure and adjacent-scale idea from the supplied code.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = _basic_conv(channels, channels, 3, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = self.weight(torch.cat([x, y], dim=1))
        fused = x * gate + y * (1.0 - gate)
        return self.refine(fused)


class _AFPNStage2(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.block0 = _basic_conv(channels[0], channels[0], 1)
        self.block1 = _basic_conv(channels[1], channels[1], 1)
        self.down_0_to_1 = _Downsample(channels[0], channels[1], 2)
        self.up_1_to_0 = _Upsample(channels[1], channels[0], 2)
        self.asf0 = _ASF2(channels[0])
        self.asf1 = _ASF2(channels[1])
        self.refine0 = nn.Sequential(_BasicBlock(channels[0]), _BasicBlock(channels[0]))
        self.refine1 = nn.Sequential(_BasicBlock(channels[1]), _BasicBlock(channels[1]))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        x0 = self.block0(x0)
        x1 = self.block1(x1)
        o0 = self.asf0(x0, self.up_1_to_0(x1))
        o1 = self.asf1(self.down_0_to_1(x0), x1)
        return self.refine0(o0), self.refine1(o1)


class _AFPNStage3(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.down_0_to_1 = _Downsample(channels[0], channels[1], 2)
        self.down_0_to_2 = _Downsample(channels[0], channels[2], 4)
        self.down_1_to_2 = _Downsample(channels[1], channels[2], 2)
        self.up_1_to_0 = _Upsample(channels[1], channels[0], 2)
        self.up_2_to_1 = _Upsample(channels[2], channels[1], 2)
        self.up_2_to_0 = _Upsample(channels[2], channels[0], 4)

        self.asf0_01 = _ASF2(channels[0])
        self.asf0 = _ASF2(channels[0])
        self.asf1_01 = _ASF2(channels[1])
        self.asf1 = _ASF2(channels[1])
        self.asf2_01 = _ASF2(channels[2])
        self.asf2 = _ASF2(channels[2])

        self.refine0 = nn.Sequential(_BasicBlock(channels[0]), _BasicBlock(channels[0]))
        self.refine1 = nn.Sequential(_BasicBlock(channels[1]), _BasicBlock(channels[1]))
        self.refine2 = nn.Sequential(_BasicBlock(channels[2]), _BasicBlock(channels[2]))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor):
        f0 = self.asf0_01(x0, self.up_1_to_0(x1))
        f0 = self.asf0(f0, self.up_2_to_0(x2))

        f1 = self.asf1_01(self.down_0_to_1(x0), x1)
        f1 = self.asf1(f1, self.up_2_to_1(x2))

        f2 = self.asf2_01(self.down_0_to_2(x0), self.down_1_to_2(x1))
        f2 = self.asf2(f2, x2)

        return self.refine0(f0), self.refine1(f1), self.refine2(f2)


class AFPN(nn.Module):
    """
    Thesis-oriented APF implementation for 3-level feature pyramids.

    The given partial code provides a generic AFPN/ASFF realization. The thesis
    only requires adjacent-level progressive fusion prior to decoupled
    distillation, so this implementation keeps that core path and adapts it to
    CIFAR-scale backbones with three semantic stages.
    """

    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("AFPN currently expects exactly three feature levels.")

        inner_channels = [max(out_channels // 2, c // 2) for c in in_channels]
        self.proj = nn.ModuleList(
            [_basic_conv(in_ch, inner_ch, 1) for in_ch, inner_ch in zip(in_channels, inner_channels)]
        )
        self.stage2 = _AFPNStage2(inner_channels[:2])
        self.stage3 = _AFPNStage3(inner_channels)
        self.out_proj = nn.ModuleList(
            [_basic_conv(inner_ch, out_channels, 1) for inner_ch in inner_channels]
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        x0, x1, x2 = [proj(feat) for proj, feat in zip(self.proj, features)]
        x0, x1 = self.stage2(x0, x1)
        x0, x1, x2 = self.stage3(x0, x1, x2)
        return [proj(feat) for proj, feat in zip(self.out_proj, [x0, x1, x2])]

