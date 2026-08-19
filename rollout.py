"""
rollout.py — 15-step open-loop rollout comparison of JEPA, PixelTransformer,
and VelocityBaseline, with position error, pixel MSE, foreground MSE, and SSIM.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import flatten_frames, load_frames, load_positions, unflatten_frames
from metrics import foreground_mse, pixel_mse, position_error, ssim, to_hwc_float
from models import PixelTransformer, SimpleJEPA, VelocityBaseline, patchify, unpatchify
from utils import ensure_dirs, get_device, load_checkpoint

NUM_STEPS = 15


def rollout_jepa(model, ctx_frame, num_steps, device):
    imgs = [ctx_frame]
    with torch.no_grad():
        z = model.encode(ctx_frame.unsqueeze(0).to(device))
        for _ in range(num_steps):
            z = model.predict(z)
            img = model.decode(z).squeeze(0).cpu()
            imgs.append(img)
    return torch.stack(imgs, dim=0)


def rollout_transformer(model, ctx_frame, num_steps, device):
    imgs = [ctx_frame]
    with torch.no_grad():
        patches = patchify(ctx_frame.unsqueeze(0).to(device))
        for _ in range(num_steps):
            patches = model(patches)
            img = unpatchify(patches).squeeze(0).cpu()
            imgs.append(img)
            patches = patchify(img.unsqueeze(0).to(device))
    return torch.stack(imgs, dim=0)


def rollout_velocity(model: VelocityBaseline, frame_t, frame_t1, num_steps):
    """Velocity baseline needs two context frames (x_t, x_t+1) to predict x_t+2, then rolls forward."""
    imgs = [frame_t, frame_t1]
    xt, xt1 = frame_t, frame_t1
    for _ in range(num_steps - 1):
        xt_flat = flatten_frames(xt.unsqueeze(0))
        xt1_flat = flatten_frames(xt1.unsqueeze(0))
        pred_flat = model.predict(xt_flat, xt1_flat)
        pred = unflatten_frames(pred_flat).squeeze(0)
        imgs.append(pred)
        xt, xt1 = xt1, pred
    return torch.stack(imgs, dim=0)


def compute_metrics_for_rollout(pred_frames, gt_frames, gt_positions, start_idx):
    """gt_positions: full (N,2,2) array; start_idx aligns pred_frames[0] with gt_frames[start_idx]."""
    steps = pred_frames.shape[0] - 1
    pos_err_red, pos_err_blue, pos_err_mean = [], [], []
    p_mse, f_mse, s_sim = [], [], []
    for t in range(1, steps + 1):
        pred_np = to_hwc_float(pred_frames[t])
        gt_np = to_hwc_float(gt_frames[t])
        gt_red = gt_positions[start_idx + t, 0]
        gt_blue = gt_positions[start_idx + t, 1]

        r_err, b_err, m_err = position_error(pred_np, gt_red, gt_blue)
        pos_err_red.append(r_err)
        pos_err_blue.append(b_err)
        pos_err_mean.append(m_err)
        p_mse.append(pixel_mse(pred_np, gt_np))
        f_mse.append(foreground_mse(pred_np, gt_np))
        s_sim.append(ssim(pred_np, gt_np))
    return {
        "pos_err_red": pos_err_red, "pos_err_blue": pos_err_blue, "pos_err_mean": pos_err_mean,
        "pixel_mse": p_mse, "foreground_mse": f_mse, "ssim": s_sim,
    }


def run(start_idx=500, num_steps=NUM_STEPS, frames_path="data/frames.npy",
        positions_path="data/positions.npy", ckpt_dir="checkpoints", out_dir="outputs"):
    ensure_dirs(out_dir)
    device = get_device()

    frames = load_frames(frames_path)
    positions = load_positions(positions_path)
    gt_frames = frames[start_idx:start_idx + num_steps + 1]

    jepa_path = os.path.join(ckpt_dir, "jepa_with_decoder.pth")
    if not os.path.exists(jepa_path):
        raise FileNotFoundError(
            f"{jepa_path} not found. Pixel-space rollout needs a trained decoder — "
            f"run train_jepa.py then train_decoder.py first. For decoder-free latent-space "
            f"evaluation, use evaluate_latent_dynamics.py instead."
        )
    jepa = SimpleJEPA(latent_dim=64).to(device)
    load_checkpoint(jepa, jepa_path, device=device)
    jepa.eval()

    trans = PixelTransformer().to(device)
    load_checkpoint(trans, os.path.join(ckpt_dir, "transformer_best.pth"), device=device)
    trans.eval()

    vel = VelocityBaseline().load(os.path.join(ckpt_dir, "velocity_baseline.pkl"))

    jepa_preds = rollout_jepa(jepa, gt_frames[0], num_steps, device)
    trans_preds = rollout_transformer(trans, gt_frames[0], num_steps, device)
    vel_preds = rollout_velocity(vel, gt_frames[0], gt_frames[1], num_steps)

    jepa_metrics = compute_metrics_for_rollout(jepa_preds, gt_frames, positions, start_idx)
    trans_metrics = compute_metrics_for_rollout(trans_preds, gt_frames, positions, start_idx)
    vel_metrics = compute_metrics_for_rollout(vel_preds, gt_frames, positions, start_idx)

    all_metrics = {"JEPA": jepa_metrics, "Pixel Transformer": trans_metrics, "Velocity Baseline": vel_metrics}
    colors = {"JEPA": "tab:blue", "Pixel Transformer": "tab:orange", "Velocity Baseline": "tab:green"}
    steps_x = list(range(1, num_steps + 1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [
        ("pos_err_mean", "Position Error (px)", axes[0, 0]),
        ("pixel_mse", "Pixel MSE (full image)", axes[0, 1]),
        ("foreground_mse", "Foreground MSE (ball bbox)", axes[1, 0]),
        ("ssim", "SSIM", axes[1, 1]),
    ]
    for key, ylabel, ax in panels:
        for name, m in all_metrics.items():
            ax.plot(steps_x, m[key], marker="o", label=name, color=colors[name], linewidth=2)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "rollout_metrics.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")

    summary_path = os.path.join(out_dir, "rollout_summary.csv")
    with open(summary_path, "w") as f:
        f.write("model,metric,mean,std,final_step\n")
        for name, m in all_metrics.items():
            for key in ("pos_err_mean", "pixel_mse", "foreground_mse", "ssim"):
                arr = np.array(m[key], dtype=np.float64)
                arr = arr[~np.isnan(arr)]
                mean = arr.mean() if len(arr) else float("nan")
                std = arr.std() if len(arr) else float("nan")
                final = m[key][-1]
                f.write(f"{name},{key},{mean:.4f},{std:.4f},{final:.4f}\n")
    print(f"Saved {summary_path}")

    torch.save(jepa_preds, os.path.join(out_dir, "jepa_rollout_frames.pt"))
    torch.save(trans_preds, os.path.join(out_dir, "transformer_rollout_frames.pt"))
    torch.save(vel_preds, os.path.join(out_dir, "velocity_rollout_frames.pt"))
    torch.save(gt_frames, os.path.join(out_dir, "gt_rollout_frames.pt"))

    print("\nRollout complete. Summary:")
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    args = parser.parse_args()
    run(start_idx=args.start_idx, num_steps=args.num_steps)
