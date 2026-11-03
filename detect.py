import argparse
import os
import glob
import cv2
import torch
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

def load_yaml_cfg(path):
    if not os.path.exists(path):
        return {}
    if yaml is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # Lightweight fallback parser
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v.startswith("[") and v.endswith("]"):
                    v = [item.strip(" '\"") for item in v[1:-1].split(",") if item.strip()]
                cfg[k] = v
    return cfg

from models import create_aglyolo_nano, create_aglyolo_small, create_aglyolo_medium
from data import FabricDataPreprocessor
from utils import postprocess_batch, plot_detection_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run AGL-YOLO Defect Detection & Attention Heatmap Inference")
    parser.add_argument("--source", type=str, default="assets/samples/sample_2.jpg",
                        help="Path to an image file or directory of images")
    parser.add_argument("--weights", type=str, default="", help="Path to checkpoint .pt file (optional)")
    parser.add_argument("--model", type=str, default="small", choices=["nano", "small", "medium"],
                        help="Model variant")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--img-size", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Display confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output-dir", type=str, default="runs/detect", help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using {device}")

    # Load classes
    class_names = ["spot"]
    if os.path.exists(args.data):
        data_cfg = load_yaml_cfg(args.data)
        if "names" in data_cfg and isinstance(data_cfg["names"], list):
            class_names = data_cfg["names"]

    # Build model
    builders = {
        "nano": create_aglyolo_nano,
        "small": create_aglyolo_small,
        "medium": create_aglyolo_medium
    }
    model = builders[args.model](num_classes=len(class_names), img_size=args.img_size).to(device)

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[Checkpoint] Loaded weights from {args.weights}")
    else:
        print("[Info] Running with initialized model architecture (for demo / structural pipeline verification).")

    model.eval()

    # Collect source images
    if os.path.isdir(args.source):
        img_paths = sorted(
            glob.glob(os.path.join(args.source, "*.jpg")) +
            glob.glob(os.path.join(args.source, "*.png")) +
            glob.glob(os.path.join(args.source, "*.jpeg"))
        )
    else:
        img_paths = [args.source]

    if not img_paths or not os.path.exists(img_paths[0]):
        raise FileNotFoundError(f"No source images found at {args.source}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Inference] Running on {len(img_paths)} image(s)...")

    for p in img_paths:
        raw_img = cv2.imread(p)
        if raw_img is None:
            continue
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # 1. CLAHE Preprocessing
        clahe_rgb = FabricDataPreprocessor.apply_clahe(raw_rgb)

        # 2. Letterbox padding
        letterboxed, _ = FabricDataPreprocessor.letterbox(clahe_rgb, np.zeros((0, 5)), args.img_size)

        tensor_in = torch.from_numpy(letterboxed).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor_in = tensor_in.to(device)

        with torch.no_grad():
            outputs = model(tensor_in)
            attn_map = model.get_attention_maps(tensor_in)[0, 0].cpu().numpy()
            preds = postprocess_batch(outputs, args.img_size,
                                      conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)[0]

        fname = os.path.basename(p)
        save_path = os.path.join(args.output_dir, f"detected_{fname}")
        plot_detection_results(letterboxed, preds, class_names, attn_map=attn_map, save_path=save_path)
        print(f"Processed {fname} -> Found {len(preds['scores'])} detections -> Saved to {save_path}")

    print(f"\n[Completed] All inference results saved in {args.output_dir}\n")


if __name__ == "__main__":
    main()
