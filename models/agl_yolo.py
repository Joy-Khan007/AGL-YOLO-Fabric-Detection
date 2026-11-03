import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import AGLYOLO_Backbone
from .neck import BiFPNFeatureFusion
from .head import LightweightSharedHead


class AGLYOLO(nn.Module):
    """AGL-YOLO Full Network Architecture.
    
    Combines:
      1. Ghost-Convolution Backbone (extracts P2, P3, P4, P5 multi-scale features)
      2. Attention Mechanisms (Coordinate Attention in P5 and P2 + SimAM capability)
      3. BiFPN Neck with Micro-Defect Layer (160x160 P2 layer with learnable weighted fusion)
      4. Lightweight Shared-Convolution Head with GroupNorm and per-scale scale layers
    """
    def __init__(self, num_classes: int = 1, width_mult: float = 1.0, img_size: int = 640):
        super(AGLYOLO, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone = AGLYOLO_Backbone(in_channels=3, width_mult=width_mult)
        b_channels = self.backbone.out_channels
        neck_channels = max(64, int(128 * width_mult))

        self.neck = BiFPNFeatureFusion(in_channels_list=b_channels, out_channels=neck_channels)
        self.head = LightweightSharedHead(num_classes=num_classes, in_channels=neck_channels,
                                           strides=[4, 8, 16, 32])

    def forward(self, x: torch.Tensor):
        b_feats = self.backbone(x)
        n_feats = self.neck(b_feats)
        outputs = self.head(n_feats)
        return outputs

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts Coordinate Attention spatial activation heatmap for interpretability."""
        b_feats = self.backbone(x)
        n_feats = self.neck(b_feats)
        p2_feat = n_feats[0]
        heatmap = torch.mean(torch.abs(p2_feat), dim=1, keepdim=True)
        heatmap = F.interpolate(heatmap, size=x.shape[2:], mode='bilinear', align_corners=False)
        min_val, max_val = heatmap.min(), heatmap.max()
        return (heatmap - min_val) / (max_val - min_val + 1e-6)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Architecture Presets
def create_aglyolo_nano(num_classes: int = 1, img_size: int = 640) -> AGLYOLO:
    """AGL-YOLO-Nano: Ultra-lightweight variant (~0.5M params) for micro-controllers / edge."""
    return AGLYOLO(num_classes=num_classes, width_mult=0.5, img_size=img_size)


def create_aglyolo_small(num_classes: int = 1, img_size: int = 640) -> AGLYOLO:
    """AGL-YOLO-Small: Primary paper architecture (1.20M params, 10.81 GFLOPs, 81.4 FPS)."""
    return AGLYOLO(num_classes=num_classes, width_mult=0.75, img_size=img_size)


def create_aglyolo_medium(num_classes: int = 1, img_size: int = 640) -> AGLYOLO:
    """AGL-YOLO-Medium: Higher capacity variant (~2.2M params) for complex multi-class defect sets."""
    return AGLYOLO(num_classes=num_classes, width_mult=1.0, img_size=img_size)
