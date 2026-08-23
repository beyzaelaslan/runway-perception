"""Eğitim döngüsü — config'ten okuyan, tekrar üretilebilir, checkpoint kaydeden.

Çalıştırma:
    python -m src.training.train --config configs/unet_r34.yaml

Smoke test (pipeline'ı hızlı doğrula, birkaç batch):
    python -m src.training.train --config configs/unet_r34.yaml --max-batches 2 --epochs 1

Cihaz otomatik seçilir: Colab'da CUDA, lokalde CPU. Asıl eğitim Colab GPU'da koşar;
lokal CPU sadece pipeline doğrulaması içindir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LardDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.factory import create_model
from src.training.losses import DiceBCELoss
from src.training.metrics import SegmentationMetrics
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def _make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Config'ten train/val DataLoader'ları kurar."""
    d = cfg["data"]
    img_size = d["image_size"]
    aug = cfg.get("augmentation", {})

    train_ds = LardDataset(d["root"], "train", get_train_transforms(
        img_size,
        hflip=aug.get("horizontal_flip", 0.5),
        brightness_contrast=aug.get("brightness_contrast", 0.3),
        rotate_limit=aug.get("rotate_limit", 7),
    ))
    val_ds = LardDataset(d["root"], "val", get_eval_transforms(img_size))

    bs = cfg["train"]["batch_size"]
    nw = d.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None,
               max_batches=None) -> tuple[float, dict]:
    """Bir epoch train (optimizer verilirse) veya eval koşar. (ort_loss, metrikler)."""
    is_train = optimizer is not None
    model.train(is_train)
    metrics = SegmentationMetrics()
    total_loss, n = 0.0, 0

    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
            else:
                logits = model(images)
                loss = loss_fn(logits, masks)

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss)
        n += 1
        metrics.update(logits.detach(), masks)

    return total_loss / max(n, 1), metrics.compute()


def train(config_path: str, epochs: int | None, max_batches: int | None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = epochs if epochs is not None else cfg["train"]["epochs"]
    print(f"Cihaz: {device} | epoch: {n_epochs}")

    train_loader, val_loader = _make_loaders(cfg)
    print(f"train batch: {len(train_loader)} | val batch: {len(val_loader)}")

    model = create_model(cfg["model"]).to(device)
    loss_cfg = cfg["train"]["loss"]
    loss_fn = DiceBCELoss(loss_cfg["dice_weight"], loss_cfg["bce_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    use_amp = cfg["train"].get("amp", True) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_iou = -1.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = _run_epoch(model, train_loader, loss_fn, device,
                                   optimizer, scaler, max_batches)
        va_loss, va_m = _run_epoch(model, val_loader, loss_fn, device,
                                   max_batches=max_batches)
        scheduler.step()

        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
               "train_iou": tr_m["iou"], "val_iou": va_m["iou"],
               "val_dice": va_m["dice"], "val_precision": va_m["precision"],
               "val_recall": va_m["recall"], "val_f1": va_m["f1"]}
        history.append(rec)
        print(f"[{epoch}/{n_epochs}] {time.time()-t0:.0f}s | "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
              f"val_IoU={va_m['iou']:.4f} val_Dice={va_m['dice']:.4f}")

        # En iyi checkpoint'i val IoU'ya göre kaydet
        if va_m["iou"] > best_iou:
            best_iou = va_m["iou"]
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_iou": best_iou},
                       out_dir / cfg["output"]["best_name"])

    if cfg["output"].get("save_last", True):
        torch.save({"model_state": model.state_dict(), "config": cfg,
                    "epoch": n_epochs}, out_dir / "last.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Bitti. En iyi val IoU: {best_iou:.4f} -> {out_dir/cfg['output']['best_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pist segmentasyonu eğitimi")
    parser.add_argument("--config", type=str, default="configs/unet_r34.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="config'i geçersiz kıl")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="epoch başına batch sınırı (smoke test için)")
    args = parser.parse_args()
    train(args.config, args.epochs, args.max_batches)


if __name__ == "__main__":
    main()
