import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import GhostConv, C2fGhost
from .attention import CoordinateAttention


class BiFPNFeatureFusion(nn.Module):
    """Weighted Bidirectional Feature Pyramid Network (BiFPN).
    
    Fuses multi-scale feature maps across P2 (160x160), P3 (80x80), P4 (40x40), and P5 (20x20).
    Unlike standard YOLO necks that only fuse P3-P5, AGL-YOLO incorporates an explicit shallow
    P2 layer (160x160) to prevent tiny spot defects (few pixels wide) from vanishing.
    """
    def __init__(self, in_channels_list: list = [64, 128, 256, 512], out_channels: int = 128):
        super(BiFPNFeatureFusion, self).__init__()
        c2, c3, c4, c5 = in_channels_list

        self.p2_proj = GhostConv(c2, out_channels, 1)
        self.p3_proj = GhostConv(c3, out_channels, 1)
        self.p4_proj = GhostConv(c4, out_channels, 1)
        self.p5_proj = GhostConv(c5, out_channels, 1)

        self.w_td4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_td3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_td2 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.node_td4 = C2fGhost(out_channels, out_channels, n=1)
        self.node_td3 = C2fGhost(out_channels, out_channels, n=1)
        self.node_td2 = C2fGhost(out_channels, out_channels, n=1)

        self.ca_attn = CoordinateAttention(out_channels, out_channels)

    def forward(self, features: list) -> list:
        p2 = self.p2_proj(features[0])
        p3 = self.p3_proj(features[1])
        p4 = self.p4_proj(features[2])
        p5 = self.p5_proj(features[3])

        # Top-down fusion P5 -> P4
        w_td4 = F.relu(self.w_td4) / (torch.sum(F.relu(self.w_td4)) + 1e-4)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = self.node_td4(w_td4[0] * p4 + w_td4[1] * p5_up)

        # Top-down fusion P4 -> P3
        w_td3 = F.relu(self.w_td3) / (torch.sum(F.relu(self.w_td3)) + 1e-4)
        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = self.node_td3(w_td3[0] * p3 + w_td3[1] * p4_up)

        # Top-down fusion P3 -> P2 (Micro-defect layer)
        w_td2 = F.relu(self.w_td2) / (torch.sum(F.relu(self.w_td2)) + 1e-4)
        p3_up = F.interpolate(p3_td, size=p2.shape[2:], mode='nearest')
        p2_out = self.node_td2(w_td2[0] * p2 + w_td2[1] * p3_up)
        p2_out = self.ca_attn(p2_out)

        return [p2_out, p3_td, p4_td, p5]
