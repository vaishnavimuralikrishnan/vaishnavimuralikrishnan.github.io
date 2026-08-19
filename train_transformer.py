"""
train_transformer.py — Train PixelTransformer as a pixel-space baseline.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import FramePatchPairDataset, load_frames, train_val_split
from models import PixelTransformer
from utils import get_device, save_checkpoint, set_seed, ensure_dirs


def run(epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, seed=0,
        frames_path="data/frames.npy", ckpt_dir="checkpoints", log_path="outputs/transformer_log.csv"):
    set_seed(seed)
    ensure_dirs(ckpt_dir, os.path.dirname(log_path) or ".")
    device = get_device()
    print(f"Using device: {device}")

    frames = load_frames(frames_path)
    train_frames, val_frames = train_val_split(frames, val_fraction=0.2)
    train_loader = DataLoader(FramePatchPairDataset(train_frames), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(FramePatchPairDataset(val_frames), batch_size=batch_size, shuffle=False)
    print(f"Train pairs: {len(train_loader.dataset)}, Val pairs: {len(val_loader.dataset)}")

    model = PixelTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    best_val_loss = float("inf")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_patches, y_patches in train_loader:
                x_patches, y_patches = x_patches.to(device), y_patches.to(device)
                optimizer.zero_grad()
                pred = model(x_patches)
                loss = criterion(pred, y_patches)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_patches, y_patches in val_loader:
                    x_patches, y_patches = x_patches.to(device), y_patches.to(device)
                    pred = model(x_patches)
                    val_loss += criterion(pred, y_patches).item()
            val_loss /= max(len(val_loader), 1)

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            print(f"[Transformer] Epoch {epoch + 1}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
            f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, os.path.join(ckpt_dir, "transformer_best.pth"))
                print(f"  -> saved new best (val {best_val_loss:.6f})")

    save_checkpoint(model, os.path.join(ckpt_dir, "transformer_final.pth"))
    print("Transformer training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed)
