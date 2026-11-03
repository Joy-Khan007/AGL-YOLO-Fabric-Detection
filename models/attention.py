import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention Module (Hou et al. 2021).
    
    Encodes spatial position along horizontal (X) and vertical (Y) axes.
    This direction-aware mechanism is critical for long, thin fabric defects (e.g. yarn-cut,
    snags) and localized spot defects by preserving precise spatial location.
    """
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class SimAM(nn.Module):
    """Parameter-free 3D Attention Module (SimAM - Yang et al. 2021).
    
    Derives 3D attention weights based on an energy function inspired by visual neuroscience,
    evaluating spatial and channel importance without introducing extra trainable parameters.
    """
    def __init__(self, e_lambda: float = 1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        n_pixels = h * w - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n_pixels
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)
