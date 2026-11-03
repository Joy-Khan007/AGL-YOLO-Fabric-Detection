import random
import cv2
import numpy as np


class FabricDataPreprocessor:
    """Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    and Photometric/Geometric Data Augmentation for Fabric Defect Images.
    """
    @staticmethod
    def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.5, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """Applies CLAHE on L-channel of LAB color space to enhance subtle defect contrast."""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced

    @staticmethod
    def letterbox(img: np.ndarray, boxes: np.ndarray, target_size: int = 640):
        """Resizes with aspect-ratio preserved padding.
        Boxes are cx, cy, w, h normalized (0-1) and remapped to the padded square canvas.
        """
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 1] = (boxes[:, 1] * w * scale + left) / target_size
            boxes[:, 2] = (boxes[:, 2] * h * scale + top) / target_size
            boxes[:, 3] = (boxes[:, 3] * w * scale) / target_size
            boxes[:, 4] = (boxes[:, 4] * h * scale) / target_size
        return canvas, boxes

    @staticmethod
    def random_hflip(img: np.ndarray, boxes: np.ndarray, p: float = 0.5):
        """Random horizontal flip with box coordinate inversion."""
        if random.random() < p:
            img = img[:, ::-1, :].copy()
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        return img, boxes

    @staticmethod
    def mosaic(quad_images: list, img_size: int):
        """Combines 4 (img, boxes) pairs into one img_size x img_size mosaic.
        
        Letterboxes each into its own quadrant (tile = img_size/2) and remaps boxes
        into full-canvas normalized coordinates. Increases effective training diversity 4x per sample.
        """
        tile = img_size // 2
        canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        offsets = [(0, 0), (tile, 0), (0, tile), (tile, tile)]
        all_boxes = []
        for (img, boxes), (ox, oy) in zip(quad_images, offsets):
            tile_img, tile_boxes = FabricDataPreprocessor.letterbox(img, boxes, tile)
            canvas[oy:oy + tile, ox:ox + tile] = tile_img
            if len(tile_boxes) > 0:
                b = tile_boxes.copy()
                b[:, 1] = (b[:, 1] * tile + ox) / img_size
                b[:, 2] = (b[:, 2] * tile + oy) / img_size
                b[:, 3] = b[:, 3] * tile / img_size
                b[:, 4] = b[:, 4] * tile / img_size
                all_boxes.append(b)
        boxes_out = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 5), dtype=np.float32)
        return canvas, boxes_out

    @staticmethod
    def random_rotate_scale(img: np.ndarray, boxes: np.ndarray, img_size: int,
                             degrees: float = 15.0, scale_range: tuple = (0.7, 1.3), p: float = 0.5):
        """Random rotation + scale jitter around canvas center, with corner transformation box remapping."""
        if random.random() > p:
            return img, boxes
        angle = random.uniform(-degrees, degrees)
        scale = random.uniform(*scale_range)
        center = (img_size / 2, img_size / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img_t = cv2.warpAffine(img, M, (img_size, img_size), borderValue=(114, 114, 114))
        if len(boxes) == 0:
            return img_t, boxes

        b = boxes.copy()
        cx, cy, w, h = b[:, 1] * img_size, b[:, 2] * img_size, b[:, 3] * img_size, b[:, 4] * img_size
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        new_rows = []
        for i in range(len(b)):
            corners = np.array([[x1[i], y1[i]], [x2[i], y1[i]], [x2[i], y2[i]], [x1[i], y2[i]]])
            ones = np.ones((4, 1))
            pts = np.hstack([corners, ones])
            transformed = (M @ pts.T).T
            nx1, ny1 = max(transformed[:, 0].min(), 0), max(transformed[:, 1].min(), 0)
            nx2, ny2 = min(transformed[:, 0].max(), img_size), min(transformed[:, 1].max(), img_size)
            if nx2 - nx1 < 2 or ny2 - ny1 < 2:
                continue
            new_rows.append([b[i, 0], (nx1 + nx2) / 2 / img_size, (ny1 + ny2) / 2 / img_size,
                             (nx2 - nx1) / img_size, (ny2 - ny1) / img_size])
        boxes_out = np.array(new_rows, dtype=np.float32) if new_rows else np.zeros((0, 5), dtype=np.float32)
        return img_t, boxes_out

    @staticmethod
    def random_hsv(img: np.ndarray, h_gain: float = 0.015, s_gain: float = 0.5, v_gain: float = 0.3, p: float = 0.5):
        """Random HSV color jitter - helps model generalize across fabric color/lighting variations."""
        if random.random() > p:
            return img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 0] = (img_hsv[..., 0] + random.uniform(-h_gain, h_gain) * 179) % 179
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * random.uniform(1 - s_gain, 1 + s_gain), 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * random.uniform(1 - v_gain, 1 + v_gain), 0, 255)
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
