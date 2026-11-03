import math
import torch
import torch.nn as nn
from .attention import CoordinateAttention


class GhostConv(nn.Module):
    """Ghost Convolution Block (Han et al. 2020).
    
    Splits output channels into primary conv features and cheap depthwise linear transform features
    to dramatically reduce computational cost and parameter count while preserving feature richness.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 ratio: int = 2, dw_kernel_size: int = 3, stride: int = 1, relu: bool = True):
        super(GhostConv, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU() if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU() if relu else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class C2fGhost(nn.Module):
    """Ghost-based Cross-Stage Partial (CSP) Bottleneck module (C2fGhost).
    
    Concatenates intermediate feature maps from Ghost modules for lightweight, rich gradient flow.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1,
                 shortcut: bool = True, g: int = 1, e: float = 0.5):
        super(C2fGhost, self).__init__()
        self.c = int(out_channels * e)
        self.cv1 = GhostConv(in_channels, 2 * self.c, kernel_size=1)
        self.cv2 = GhostConv((2 + n) * self.c, out_channels, kernel_size=1)
        self.m = nn.ModuleList([GhostConv(self.c, self.c, kernel_size=3) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AGLYOLO_Backbone(nn.Module):
    """Ghost-convolution backbone extracting multi-scale features at P2, P3, P4, and P5.
    
    Features:
      - Stem: 2 GhostConvs with initial stride 2 downsampling
      - Stage P2 (stride 4, 160x160): Shallow high-resolution features for tiny defects
      - Stage P3 (stride 8, 80x80): Medium feature representation
      - Stage P4 (stride 16, 40x40): Semantic feature representation
      - Stage P5 (stride 32, 20x20): Deep representation enhanced with Coordinate Attention
    """
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super(AGLYOLO_Backbone, self).__init__()
        c = lambda ch: max(16, int(ch * width_mult))

        self.stem = nn.Sequential(
            GhostConv(in_channels, c(16), 3, stride=2),
            GhostConv(c(16), c(32), 3, stride=1)
        )
        self.stage_p2 = nn.Sequential(
            GhostConv(c(32), c(64), 3, stride=2),
            C2fGhost(c(64), c(64), n=1)
        )
        self.stage_p3 = nn.Sequential(
            GhostConv(c(64), c(128), 3, stride=2),
            C2fGhost(c(128), c(128), n=2)
        )
        self.stage_p4 = nn.Sequential(
            GhostConv(c(128), c(256), 3, stride=2),
            C2fGhost(c(256), c(256), n=2)
        )
        self.stage_p5 = nn.Sequential(
            GhostConv(c(256), c(512), 3, stride=2),
            C2fGhost(c(512), c(512), n=1),
            CoordinateAttention(c(512), c(512))
        )
        self.out_channels = [c(64), c(128), c(256), c(512)]

    def forward(self, x: torch.Tensor) -> list:
        x = self.stem(x)
        p2 = self.stage_p2(x)
        p3 = self.stage_p3(p2)
        p4 = self.stage_p4(p3)
        p5 = self.stage_p5(p4)
        return [p2, p3, p4, p5]
