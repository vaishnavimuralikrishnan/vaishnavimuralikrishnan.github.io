"""
train_velocity_baseline.py — Fit a simple linear-regression sanity-check baseline:
    concat(x_t, velocity_t) -> x_t+2,  where velocity_t = x_t+1 - x_t
"""
import argparse
import os

import numpy as np

from data import FrameTripletDataset, flatten_frames, load_frames, train_val_split
from models import VelocityBaseline
from utils import ensure_dirs, set_seed


def stack_dataset(dataset: FrameTripletDataset, max_samples: int | None = None):
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    xt, xt1, xt2 = [], [], []
    for i in range(n):
        a, b, c = dataset[i]
        xt.append(a)
        xt1.append(b)
        xt2.append(c)
    import torch
    return torch.stack(xt), torch.stack(xt1), torch.stack(xt2)


def run(frames_path="data/frames.npy", ckpt_dir="checkpoints", seed=0, max_train_samples=4000):
    set_seed(seed)
    ensure_dirs(ckpt_dir)

    frames = load_frames(frames_path)
    train_frames, val_frames = train_val_split(frames, val_fraction=0.2)

    train_ds = FrameTripletDataset(train_frames)
    val_ds = FrameTripletDataset(val_frames)

    print("Preparing training features (subsampled for tractable linear regression)...")
    xt, xt1, xt2 = stack_dataset(train_ds, max_samples=max_train_samples)
    xt_flat, xt1_flat, xt2_flat = flatten_frames(xt), flatten_frames(xt1), flatten_frames(xt2)

    model = VelocityBaseline()
    model.fit(xt_flat, xt1_flat, xt2_flat)

    pred_train = model.predict(xt_flat, xt1_flat)
    train_mse = float(np.mean((pred_train.numpy() - xt2_flat.numpy()) ** 2))

    xt_v, xt1_v, xt2_v = stack_dataset(val_ds, max_samples=max_train_samples // 4)
    xt_v_flat, xt1_v_flat, xt2_v_flat = flatten_frames(xt_v), flatten_frames(xt1_v), flatten_frames(xt2_v)
    pred_val = model.predict(xt_v_flat, xt1_v_flat)
    val_mse = float(np.mean((pred_val.numpy() - xt2_v_flat.numpy()) ** 2))

    print(f"Velocity baseline — Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

    out_path = os.path.join(ckpt_dir, "velocity_baseline.pkl")
    model.save(out_path)
    print(f"Saved fitted model -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    args = parser.parse_args()
    run(seed=args.seed, max_train_samples=args.max_train_samples)
