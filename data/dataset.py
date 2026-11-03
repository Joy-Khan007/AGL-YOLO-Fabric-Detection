import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessor import FabricDataPreprocessor


def find_label_path(img_path: str) -> str:
    """Converts YOLO convention image path to label path.
    Example: .../images/sample.jpg -> .../labels/sample.txt
    """
    normalized = img_path.replace('\\', '/')
    if '/images/' in normalized:
        return normalized.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    
    # Fallback if flat directory
    parent, fname = os.path.split(img_path)
    label_dir = os.path.join(os.path.dirname(parent), 'labels') if os.path.basename(parent) == 'images' else parent
    return os.path.join(label_dir, fname.rsplit('.', 1)[0] + '.txt')


class YOLOFabricDataset(Dataset):
    """Loads real fabric images + YOLO-format bounding box labels.
    
    Training pipeline:
      CLAHE -> Mosaic (4-image combine) -> Random Rotation / Scale ->
      Random Horizontal Flip -> Random HSV Jitter -> Pixel Coordinates
      
    Validation pipeline:
      CLAHE -> Aspect-preserved Letterbox -> Normalized to Pixel Coordinates
    """
    def __init__(self, img_dir: str, img_size: int = 640, augment: bool = True, mosaic_prob: float = 0.5):
        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.jpeg')) +
            glob.glob(os.path.join(img_dir, '*.png'))
        )
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}. Check dataset paths.")
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0

    def __len__(self):
        return len(self.img_paths)

    def _load_labels(self, label_path: str) -> np.ndarray:
        if not os.path.exists(label_path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(label_path, 'r') as f:
            for line in f.read().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls_id, cx, cy, w, h = (float(x) for x in parts[:5])
                rows.append([cls_id, cx, cy, w, h])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)

    def _load_raw(self, idx: int):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self._load_labels(find_label_path(img_path))
        img = FabricDataPreprocessor.apply_clahe(img)
        return img, boxes

    def __getitem__(self, idx: int):
        if random.random() < self.mosaic_prob:
            idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            quad = [self._load_raw(i) for i in idxs]
            img, boxes = FabricDataPreprocessor.mosaic(quad, self.img_size)
        else:
            img, boxes = self._load_raw(idx)
            img, boxes = FabricDataPreprocessor.letterbox(img, boxes, self.img_size)

        if self.augment:
            img, boxes = FabricDataPreprocessor.random_rotate_scale(img, boxes, self.img_size)
            img, boxes = FabricDataPreprocessor.random_hflip(img, boxes)
            img = FabricDataPreprocessor.random_hsv(img)

        # Convert normalized cx,cy,w,h -> pixel xyxy for loss calculation
        if len(boxes) > 0:
            cls_id = boxes[:, 0]
            cx = boxes[:, 1] * self.img_size
            cy = boxes[:, 2] * self.img_size
            w = boxes[:, 3] * self.img_size
            h = boxes[:, 4] * self.img_size
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            target = np.stack([cls_id, x1, y1, x2, y2], axis=1)
        else:
            target = np.zeros((0, 5), dtype=np.float32)

        tensor_in = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor_in, torch.from_numpy(target).float(), img


def yolo_collate_fn(batch):
    """Custom collate function since detection ground-truth box count varies per image."""
    imgs, targets, raws = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets), list(raws)
