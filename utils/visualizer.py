import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def plot_detection_results(image_rgb: np.ndarray, preds: dict, class_names: list,
                           attn_map: np.ndarray = None, save_path: str = None,
                           gt_boxes: np.ndarray = None):
    """Visualizes detection results with side-by-side comparison:
      1. Raw / Input Image
      2. Coordinate Attention Heatmap (if provided)
      3. Predicted Bounding Boxes + Confidence Scores
    """
    n_cols = 3 if attn_map is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), dpi=150)

    # 1. Input Sample
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Fabric Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    col = 1
    # 2. Attention Heatmap
    if attn_map is not None:
        axes[col].imshow(image_rgb)
        axes[col].imshow(attn_map, cmap='jet', alpha=0.55)
        axes[col].set_title("Coordinate Attention Heatmap", fontsize=12, fontweight='bold')
        axes[col].axis('off')
        col += 1

    # 3. Detection Predictions
    axes[col].imshow(image_rgb)
    boxes = preds.get('boxes', [])
    scores = preds.get('scores', [])
    labels = preds.get('labels', [])

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#EF4444', facecolor='none')
        axes[col].add_patch(rect)
        cls_name = class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
        axes[col].text(x1, max(y1 - 6, 12), f"{cls_name}: {score:.2f}",
                       color='white', fontsize=9, fontweight='bold',
                       bbox=dict(facecolor='#EF4444', edgecolor='none', pad=2))

    if gt_boxes is not None and len(gt_boxes) > 0:
        for gt in gt_boxes:
            gx1, gy1, gx2, gy2 = gt[1:5]
            rect_gt = patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                         linewidth=1.5, edgecolor='#10B981', linestyle='--', facecolor='none')
            axes[col].add_patch(rect_gt)

    axes[col].set_title(f"Detections ({len(scores)} found)", fontsize=12, fontweight='bold')
    axes[col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()
