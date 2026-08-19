"""
train_decoder.py — Train the decoder on FROZEN JEPA representations, for
visualization purposes only.

This is a separate, second training stage, run AFTER train_jepa.py. It is
standard practice for JEPA-style models (e.g. the visualizations in the
I-JEPA paper are produced this way): the encoder and predictor are frozen —
they never see a gradient here — and only the decoder is optimized to invert
the already-learned latent space:

    z = encoder(x)          [frozen, no grad]
    x_hat = decoder(z)
    loss = MSE(x_hat, x)

This keeps train_jepa.py's objective honest (pure latent prediction) while
still giving you a decoder you can use for find_position_dimension.py,
rollout.py, intervene.py, and the GIFs/figures that need actual pixels.

Loads checkpoints/jepa_best.pth, trains only the decoder, and saves the FULL
model (frozen encoder/predictor + newly-trained decoder) to
checkpoints/jepa_with_decoder.pth.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import SingleFrameDataset, load_frames, train_val_split
from models import SimpleJEPA
from utils import ensure_dirs, get_device, load_checkpoint, save_checkpoint, set_seed


def run(epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, seed=0,
        frames_path="data/frames.npy", ckpt_dir="checkpoints",
        jepa_ckpt="jepa_best.pth", log_path="outputs/decoder_log.csv"):
    set_seed(seed)
    ensure_dirs(ckpt_dir, os.path.dirname(log_path) or ".")
    device = get_device()
    print(f"Using device: {device}")

    jepa_path = os.path.join(ckpt_dir, jepa_ckpt)
    if not os.path.exists(jepa_path):
        raise FileNotFoundError(f"{jepa_path} not found — run train_jepa.py first.")

    model = SimpleJEPA(latent_dim=64).to(device)
    load_checkpoint(model, jepa_path, device=device)
    print(f"Loaded frozen JEPA encoder/predictor from {jepa_path}")

    # Freeze everything except the decoder.
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(False)
    for p in model.momentum_encoder.parameters():
        p.requires_grad_(False)
    model.encoder.eval()
    model.predictor.eval()
    model.momentum_encoder.eval()

    frames = load_frames(frames_path)
    train_frames, val_frames = train_val_split(frames, val_fraction=0.2)
    train_loader = DataLoader(SingleFrameDataset(train_frames), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(SingleFrameDataset(val_frames), batch_size=batch_size,
                            shuffle=False)
    print(f"Train frames: {len(train_loader.dataset)}, Val frames: {len(val_loader.dataset)}")

    optimizer = AdamW(model.decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    print(f"Decoder params (trained): {sum(p.numel() for p in model.decoder.parameters()):,}")

    best_val_loss = float("inf")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(epochs):
            model.decoder.train()
            train_loss = 0.0
            for x in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    z = model.encoder(x)  # frozen encoder, no grad
                x_hat = model.decoder(z)
                loss = criterion(x_hat, x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.decoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    z = model.encoder(x)
                    x_hat = model.decoder(z)
                    val_loss += criterion(x_hat, x).item()
            val_loss /= max(len(val_loader), 1)

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            print(f"[Decoder] Epoch {epoch + 1}/{epochs} | train {train_loss:.6f} | "
                  f"val {val_loss:.6f} | lr {current_lr:.2e}")
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
            f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, os.path.join(ckpt_dir, "jepa_with_decoder.pth"))
                print(f"  -> saved new best (val {best_val_loss:.6f})")

    print("\nDecoder training complete.")
    print(f"Full model (frozen encoder/predictor + trained decoder) saved to "
          f"{os.path.join(ckpt_dir, 'jepa_with_decoder.pth')}")
    print("Use this checkpoint (not jepa_best.pth) for find_position_dimension.py, "
          "rollout.py, intervene.py, and create_report_figures.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the JEPA decoder on frozen encoder representations (visualization only)."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jepa-ckpt", type=str, default="jepa_best.pth")
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, jepa_ckpt=args.jepa_ckpt)
