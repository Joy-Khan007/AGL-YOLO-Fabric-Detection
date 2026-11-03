import torch
import torchvision
from models.head import decode_predictions


def postprocess_batch(head_outputs, img_size: int, conf_thresh: float = 0.25,
                      iou_thresh: float = 0.45, max_det: int = 300):
    """Post-processes raw model outputs into final bounding boxes using NMS.
    
    Args:
        head_outputs: Output list from LightweightSharedHead across scales
        img_size: Image input size (e.g. 640)
        conf_thresh: Score threshold (0.25 for display/inference; use 0.001 for mAP evaluation!)
        iou_thresh: IoU overlap threshold for Non-Maximum Suppression (NMS)
        max_det: Maximum detections per image
        
    Returns:
        List of dicts per image:
          {'boxes': Tensor(N, 4), 'scores': Tensor(N,), 'labels': Tensor(N,)}
    """
    decoded = decode_predictions(head_outputs, img_size)
    B = head_outputs[0]['cls'].shape[0]
    results = []
    
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for scale in decoded:
            obj = torch.sigmoid(scale['obj'][b])
            cls_probs = torch.sigmoid(scale['cls'][b])
            scores_per_cls, labels_per_cls = cls_probs.max(dim=1)
            scores = obj * scores_per_cls
            keep = scores > conf_thresh
            if keep.sum() == 0:
                continue
            all_boxes.append(scale['boxes'][b][keep])
            all_scores.append(scores[keep])
            all_labels.append(labels_per_cls[keep])

        if len(all_boxes) == 0:
            results.append({
                'boxes': torch.zeros((0, 4)),
                'scores': torch.zeros((0,)),
                'labels': torch.zeros((0,), dtype=torch.long)
            })
            continue

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        
        final_boxes, final_scores, final_labels = [], [], []
        for c in labels.unique():
            m = labels == c
            keep_idx = torchvision.ops.nms(boxes[m], scores[m], iou_thresh)
            final_boxes.append(boxes[m][keep_idx])
            final_scores.append(scores[m][keep_idx])
            final_labels.append(labels[m][keep_idx])
            
        boxes = torch.cat(final_boxes)
        scores = torch.cat(final_scores)
        labels = torch.cat(final_labels)
        
        if len(scores) > max_det:
            top = scores.topk(max_det).indices
            boxes, scores, labels = boxes[top], scores[top], labels[top]
            
        results.append({'boxes': boxes.cpu(), 'scores': scores.cpu(), 'labels': labels.cpu()})
        
    return results
