import argparse
import os
import torch
from torch.utils.data import DataLoader

from models import create_aglyolo_nano, create_aglyolo_small, create_aglyolo_medium
from data import YOLOFabricDataset, yolo_collate_fn
from utils import evaluate_map

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

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained AGL-YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="small", choices=["nano", "small", "medium"],
                        help="AGL-YOLO variant")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--conf-thresh", type=float, default=0.001,
                        help="Confidence threshold for mAP evaluation (keep near 0 for full PR-curve)")
    parser.add_argument("--iou-thresh", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using {device}")

    # Load configuration
    data_cfg = load_yaml_cfg(args.data)

    val_img_dir = data_cfg.get("val", "")
    class_names = data_cfg.get("names", ["spot"])
    num_classes = len(class_names)

    val_ds = YOLOFabricDataset(val_img_dir, img_size=args.img_size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=yolo_collate_fn, num_workers=2)
    print(f"[Dataset] Evaluating on {len(val_ds)} validation images | Classes: {class_names}")

    # Build and load model
    builders = {
        "nano": create_aglyolo_nano,
        "small": create_aglyolo_small,
        "medium": create_aglyolo_medium
    }
    model = builders[args.model](num_classes=num_classes, img_size=args.img_size).to(device)

    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Checkpoint] Successfully loaded weights from {args.weights}")
    else:
        print(f"[Warning] Checkpoint {args.weights} not found. Evaluating with random initialization.")

    # Model complexity
    param_count = model.num_parameters()
    print(f"[Parameters] Total trainable parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    if HAS_THOP:
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"[Computation] GFLOPs: {flops / 1e9:.2f} @ {args.img_size}x{args.img_size}")

    # Evaluation
    metrics = evaluate_map(model, val_loader, args.img_size, device,
                           eval_conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)

    print("\n" + "=" * 45)
    print("        AGL-YOLO Evaluation Metrics")
    print("=" * 45)
    print(f"  mAP@0.5      : {metrics['map_50'] * 100:.2f}%")
    print(f"  mAP@0.5:0.95 : {metrics['map'] * 100:.2f}%")
    print(f"  mAR@100      : {metrics['mar_100'] * 100:.2f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
