import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import GhostConv


class LightweightSharedHead(nn.Module):
    """Lightweight Shared-Convolution Head across multi-scale feature outputs (P2, P3, P4, P5).
    
    Uses shared GhostConvolutions and Group Normalization across scale heads, plus learnable
    per-scale distance scaling layers (FCOS style) to keep parameter count minimal.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 128, strides: list = [4, 8, 16, 32]):
        super(LightweightSharedHead, self).__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.shared_cls_conv = nn.Sequential(
            GhostConv(in_channels, in_channels, kernel_size=3),
            nn.GroupNorm(8, in_channels),
            nn.SiLU()
        )
        self.shared_reg_conv = nn.Sequential(
            GhostConv(in_channels, in_channels, kernel_size=3),
            nn.GroupNorm(8, in_channels),
            nn.SiLU()
        )

        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.obj_head = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Learnable per-scale distance scale
        self.scale_layers = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in strides])

    def forward(self, features: list) -> list:
        outputs = []
        for i, feat in enumerate(features):
            cls_feat = self.shared_cls_conv(feat)
            reg_feat = self.shared_reg_conv(feat)

            cls_out = self.cls_head(cls_feat)
            reg_out = self.reg_head(reg_feat) * self.scale_layers[i]
            obj_out = self.obj_head(reg_feat)

            outputs.append({
                'cls': cls_out,   # (B, num_classes, H, W) raw logits
                'reg': reg_out,   # (B, 4, H, W) raw (l, t, r, b) distances in stride units (pre-exp)
                'obj': obj_out,   # (B, 1, H, W) raw objectness logits
                'stride': self.strides[i]
            })
        return outputs


def compute_siou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    """SIoU (SCYLLA-IoU) Loss (Zheng et al. 2022). Boxes are (x1, y1, x2, y2).
    
    Incorporates:
      1. Angle cost: Penalizes angle difference between box centers
      2. Distance cost: Normalized center point distance
      3. Shape cost: Aspect ratio and scale alignment
      4. IoU cost: Overlap loss
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) + 1e-7
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) + 1e-7

    px, py = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    tx, ty = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0

    sigma = torch.pow((tx - px) ** 2 + (ty - py) ** 2, 0.5) + 1e-7
    sin_alpha = torch.abs(ty - py) / sigma
    sin_beta = torch.abs(tx - px) / sigma
    sin_alpha = torch.where(sin_alpha > math.sin(math.pi / 4), sin_beta, sin_alpha)

    angle_cost = 1.0 - 2.0 * (torch.pow(torch.sin(torch.arcsin(sin_alpha.clamp(-1 + 1e-7, 1 - 1e-7)) - math.pi / 4), 2))
    rho_x = ((tx - px) / cw) ** 2
    rho_y = ((ty - py) / ch) ** 2
    gamma = 2.0 - angle_cost
    distance_cost = 2.0 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter / (area1 + area2 - inter + 1e-7)

    siou = iou - 0.5 * (distance_cost + 0.5)
    return 1.0 - siou.clamp(min=-1.0, max=1.0), iou.detach()


def decode_predictions(head_outputs: list, img_size: int):
    """Converts every scale's raw (l,t,r,b) distances into absolute xyxy boxes in pixel space."""
    decoded = []
    for out in head_outputs:
        B, C, H, W = out['cls'].shape
        stride = out['stride']
        yv, xv = torch.meshgrid(torch.arange(H, device=out['cls'].device),
                                torch.arange(W, device=out['cls'].device), indexing='ij')
        grid_xy = (torch.stack([xv, yv], dim=-1).float() + 0.5) * stride
        grid_xy_flat = grid_xy.view(-1, 2)

        reg = out['reg'].permute(0, 2, 3, 1).reshape(B, H * W, 4)
        ltrb = F.softplus(reg) * stride

        cx = grid_xy_flat[None, :, 0]
        cy = grid_xy_flat[None, :, 1]
        x1 = cx - ltrb[..., 0]
        y1 = cy - ltrb[..., 1]
        x2 = cx + ltrb[..., 2]
        y2 = cy + ltrb[..., 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, img_size)

        obj = out['obj'].permute(0, 2, 3, 1).reshape(B, H * W)
        cls = out['cls'].permute(0, 2, 3, 1).reshape(B, H * W, C)

        decoded.append({'boxes': boxes, 'obj': obj, 'cls': cls, 'grid_xy': grid_xy_flat, 'stride': stride})
    return decoded


class AGLYOLOLoss(nn.Module):
    """Composite Anchor-Free Loss for AGL-YOLO:
      - Center-sampling assignment of each GT box to grid cells at the best FPN scale
      - SIoU loss for box regression (positives only)
      - Focal loss for objectness (all grid cells, addressing foreground/background imbalance)
      - BCE loss for classification (positives only)
    """
    def __init__(self, num_classes: int = 1, img_size: int = 640, center_radius: float = 2.5,
                 obj_weight: float = 1.0, box_weight: float = 5.0, cls_weight: float = 1.0,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.center_radius = center_radius
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_bce(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.focal_gamma * ce
        return loss

    def forward(self, head_outputs: list, targets: list):
        """targets: list (len B) of Tensor (N_i, 5) = [cls_id, x1, y1, x2, y2] in PIXEL coordinates."""
        decoded = decode_predictions(head_outputs, self.img_size)
        device = head_outputs[0]['cls'].device
        B = head_outputs[0]['cls'].shape[0]

        total_obj_loss = torch.zeros((), device=device)
        total_box_loss = torch.zeros((), device=device)
        total_cls_loss = torch.zeros((), device=device)
        num_pos_total = 0

        for scale in decoded:
            grid_xy = scale['grid_xy']
            stride = scale['stride']
            HW = grid_xy.shape[0]

            obj_targets = torch.zeros((B, HW), device=device)
            cls_targets = torch.zeros((B, HW, self.num_classes), device=device)
            box_targets = torch.zeros((B, HW, 4), device=device)
            pos_masks = torch.zeros((B, HW), dtype=torch.bool, device=device)

            for b in range(B):
                gt = targets[b]
                if gt.numel() == 0:
                    continue
                gt_cls = gt[:, 0].long()
                gt_boxes = gt[:, 1:5]
                gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                gt_scale_metric = torch.sqrt(gt_w * gt_h)

                lo, hi = stride * 1.0, stride * 8.0
                if stride == min(s['stride'] for s in decoded):
                    lo = 0.0
                if stride == max(s['stride'] for s in decoded):
                    hi = float('inf')
                scale_ok = (gt_scale_metric >= lo) & (gt_scale_metric < hi)
                if scale_ok.sum() == 0:
                    continue

                idxs = scale_ok.nonzero(as_tuple=True)[0]
                for gi in idxs.tolist():
                    cx, cy = gt_cx[gi], gt_cy[gi]
                    radius = self.center_radius * stride
                    dist = ((grid_xy[:, 0] - cx) ** 2 + (grid_xy[:, 1] - cy) ** 2).sqrt()
                    inside = (dist < radius) & \
                             (grid_xy[:, 0] > gt_boxes[gi, 0]) & (grid_xy[:, 0] < gt_boxes[gi, 2]) & \
                             (grid_xy[:, 1] > gt_boxes[gi, 1]) & (grid_xy[:, 1] < gt_boxes[gi, 3])
                    if inside.sum() == 0:
                        continue
                    obj_targets[b, inside] = 1.0
                    cls_targets[b, inside, gt_cls[gi]] = 1.0
                    box_targets[b, inside] = gt_boxes[gi]
                    pos_masks[b, inside] = True

            obj_loss = self.focal_bce(scale['obj'], obj_targets).mean()
            total_obj_loss = total_obj_loss + obj_loss

            n_pos = int(pos_masks.sum().item())
            num_pos_total += n_pos
            if n_pos > 0:
                pred_boxes_pos = scale['boxes'][pos_masks]
                target_boxes_pos = box_targets[pos_masks]
                siou, _ = compute_siou_loss(pred_boxes_pos, target_boxes_pos)
                total_box_loss = total_box_loss + siou.mean()

                cls_logits_pos = scale['cls'][pos_masks]
                cls_targets_pos = cls_targets[pos_masks]
                total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                    cls_logits_pos, cls_targets_pos, reduction='mean')

        total_loss = (self.obj_weight * total_obj_loss +
                      self.box_weight * total_box_loss +
                      self.cls_weight * total_cls_loss)
        return total_loss, {
            'obj_loss': total_obj_loss.item(),
            'box_loss': total_box_loss.item(),
            'cls_loss': total_cls_loss.item(),
            'num_positives': num_pos_total
        }
