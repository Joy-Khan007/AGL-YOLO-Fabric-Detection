import os
import argparse
import math
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

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

from models import (
    create_aglyolo_nano,
    create_aglyolo_small,
    create_aglyolo_medium,
    AGLYOLOLoss
)
from data import YOLOFabricDataset, yolo_collate_fn
from utils import evaluate_map


def parse_args():
    parser = argparse.ArgumentParser(description="Train AGL-YOLO on Fabric Defect Dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="small", choices=["nano", "small", "medium"],
                        help="AGL-YOLO variant")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Input image dimension")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate mAP every N epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (eval cycles)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using {device}")

    # Load data configuration
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml not found at {args.data}")
    data_cfg = load_yaml_cfg(args.data)

    train_img_dir = data_cfg.get("train", "")
    val_img_dir = data_cfg.get("val", "")
    class_names = data_cfg.get("names", ["spot"])
    num_classes = len(class_names)
    print(f"[Config] Classes: {class_names} ({num_classes})")

    # Datasets and Loaders
    train_ds = YOLOFabricDataset(train_img_dir, img_size=args.img_size, augment=True)
    val_ds = YOLOFabricDataset(val_img_dir, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=yolo_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=yolo_collate_fn, num_workers=2)
    print(f"[Dataset] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model variant selection
    builders = {
        "nano": create_aglyolo_nano,
        "small": create_aglyolo_small,
        "medium": create_aglyolo_medium
    }
    model = builders[args.model](num_classes=num_classes, img_size=args.img_size).to(device)
    param_count = model.num_parameters()
    print(f"[Model Stats] Variant: AGL-YOLO-{args.model} | Trainable Params: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Loss & Optimizer
    criterion = AGLYOLOLoss(num_classes=num_classes, img_size=args.img_size, center_radius=2.5,
                            obj_weight=1.0, box_weight=5.0, cls_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine scheduler with warmup
    def get_lr(epoch):
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * progress))

    os.makedirs(args.save_dir, exist_ok=True)
    best_map50 = 0.0
    no_improve_count = 0
    best_ckpt_path = os.path.join(args.save_dir, f"aglyolo_{args.model}_best.pt")

    print(f"\n[Training Started] Training {args.model} for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        # Update learning rate
        current_lr = get_lr(epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        model.train()
        total_loss, obj_accum, box_accum, cls_accum = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)

        for imgs, targets, _ in pbar:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            outputs = model(imgs)
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            obj_accum += loss_dict["obj_loss"]
            box_accum += loss_dict["box_loss"]
            cls_accum += loss_dict["cls_loss"]
            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "LR": f"{current_lr:.6f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch:03d}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

        # Validation every N epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            try:
                metrics = evaluate_map(model, val_loader, args.img_size, device, eval_conf_thresh=0.001)
                val_map50 = metrics["map_50"]
                print(f"  -> Val mAP@0.5: {val_map50 * 100:.2f}% | mAP@0.5:0.95: {metrics['map'] * 100:.2f}%")

                if val_map50 > best_map50:
                    best_map50 = val_map50
                    no_improve_count = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"  -> [New Best] Saved checkpoint to {best_ckpt_path}")
                else:
                    no_improve_count += 1
                    if no_improve_count >= args.patience:
                        print(f"[Early Stopping] No mAP improvement for {args.patience * args.eval_every} epochs. Stopping.")
                        break
            except Exception as e:
                print(f"  -> Validation warning: {e}")

    print(f"\n[Training Finished] Best Val mAP@0.5: {best_map50 * 100:.2f}% | Checkpoint: {best_ckpt_path}\n")


if __name__ == "__main__":
    main()
