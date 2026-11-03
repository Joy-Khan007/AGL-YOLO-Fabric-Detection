import torch
from .postprocess import postprocess_batch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


@torch.no_grad()
def evaluate_map(model, loader, img_size: int, device: torch.device,
                 eval_conf_thresh: float = 0.001, iou_thresh: float = 0.45,
                 desc: str = "Evaluating mAP"):
    """Computes mAP@0.5, mAP@0.5:0.95, and mAR@100.
    
    CRITICAL LESSON FROM EXPERIMENT 1 vs 2:
      Always evaluate with a near-zero confidence threshold (e.g. 0.001) during mAP calculation.
      Filtering at conf_thresh=0.25 BEFORE computing mAP discards the tail of the precision-recall
      curve and artificially destroys the measured mAP score. The 0.25 threshold should only be used
      for visual display and live deployment!
    """
    if not HAS_TORCHMETRICS:
        raise ImportError(
            "torchmetrics is required for evaluation. Install it via: pip install torchmetrics"
        )

    was_training = model.training
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)

    for imgs, targets, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = postprocess_batch(outputs, img_size, conf_thresh=eval_conf_thresh, iou_thresh=iou_thresh)

        preds_formatted = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': p['labels']} for p in preds]
        gts_formatted = []
        for t in targets:
            if t.numel() == 0:
                gts_formatted.append({'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.long)})
            else:
                gts_formatted.append({'boxes': t[:, 1:5].cpu(), 'labels': t[:, 0].long().cpu()})
                
        metric.update(preds_formatted, gts_formatted)

    result = metric.compute()
    if was_training:
        model.train()
        
    return {
        'map_50': result['map_50'].item() if 'map_50' in result else 0.0,
        'map': result['map'].item() if 'map' in result else 0.0,
        'mar_100': result['mar_100'].item() if 'mar_100' in result else 0.0
    }
