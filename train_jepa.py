"""
train_jepa.py — Train SimpleJEPA. PURE JEPA: latent-space loss only.

This is the correct JEPA training objective:
    z_t          = encoder(x_t)
    z_next_pred  = predictor(z_t)
    z_t1_target  = momentum_encoder(x_t+1)   [no grad, EMA-updated target]
    loss         = MSE(z_next_pred, z_t1_target)

The decoder is NOT part of this loss and receives NO gradients here — this
matches the spec exactly ("Decoder ... for visualization only, NOT used in
training loss"). After this script, model.decoder is still at its random
initialization. To get a decoder that produces recognizable images (needed
for find_position_dimension.py, rollout.py, intervene.py), run
train_decoder.py afterward — it trains ONLY the decoder, on frozen,
already-trained encoder representations, and never touches the encoder or
predictor. This two-stage split keeps the JEPA training itself honest while
still giving you a working visualization tool.

If you only care about latent-space dynamics (the "real" JEPA evaluation),
see evaluate_latent_dynamics.py, which needs no decoder at all.
"""
import argparse
import csv
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import FramePairDataset, load_frames, train_val_split
from models import SimpleJEPA
from utils import get_device, save_checkpoint, set_seed, ensure_dirs


def run(epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, seed=0,
        frames_path="data/frames.npy", ckpt_dir="checkpoints",
        log_path="outputs/jepa_log.csv"):
    set_seed(seed)
    ensure_dirs(ckpt_dir, os.path.dirname(log_path) or ".")
    device = get_device()
    print(f"Using device: {device}")
    print("Training PURE JEPA: latent-space loss only, decoder is not trained here.")

    frames = load_frames(frames_path)
    train_frames, val_frames = train_val_split(frames, val_fraction=0.2)
    train_loader = DataLoader(FramePairDataset(train_frames), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(FramePairDataset(val_frames), batch_size=batch_size,
                            shuffle=False)
    print(f"Train pairs: {len(train_loader.dataset)}, Val pairs: {len(val_loader.dataset)}")

    model = SimpleJEPA(latent_dim=64, momentum_tau=0.99).to(device)

    # Only encoder + predictor are optimized. The decoder is deliberately
    # excluded from this optimizer — it gets zero gradient from this loss.
    optimizer = AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    n_trainable = (sum(p.numel() for p in model.encoder.parameters())
                   + sum(p.numel() for p in model.predictor.parameters()))
    print(f"Encoder+Predictor params (trained): {n_trainable:,}")

    best_val_loss = float("inf")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_t, x_t1 in train_loader:
                x_t, x_t1 = x_t.to(device), x_t1.to(device)
                optimizer.zero_grad()

                # ===== The ONLY loss term in JEPA =====
                loss, z_t, z_next_pred = model(x_t, x_t1)

                loss.backward()
                optimizer.step()
                model.update_momentum_encoder()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_t, x_t1 in val_loader:
                    x_t, x_t1 = x_t.to(device), x_t1.to(device)
                    loss, _, _ = model(x_t, x_t1)
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            print(f"[JEPA] Epoch {epoch + 1}/{epochs} | train {train_loss:.6f} | "
                  f"val {val_loss:.6f} | lr {current_lr:.2e}")
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
            f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, os.path.join(ckpt_dir, "jepa_best.pth"))
                print(f"  -> saved new best (val {best_val_loss:.6f})")

    save_checkpoint(model, os.path.join(ckpt_dir, "jepa_final.pth"))
    print("\nJEPA training complete (encoder + predictor only).")
    print("Decoder is still randomly initialized. Run train_decoder.py next if you")
    print("need pixel-space visualizations (find_position_dimension.py, rollout.py, intervene.py).")
    print("For pure latent-space evaluation, run evaluate_latent_dynamics.py — it needs no decoder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pure JEPA (latent-space loss only).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed)
