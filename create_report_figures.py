"""
create_report_figures.py — Final publication-quality figures for the report:

  Figure 1: rollout metrics (position error, pixel MSE, foreground MSE, SSIM),
            averaged with error bars over 5 random starting frames.
  Figure 2: disentanglement MI heatmap (delegates to validate_disentanglement.py).
  Figure 3: intervention trajectory plot (normal vs. intervened).

All figures saved as 300 DPI PNGs in outputs/.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

from data import load_frames, load_positions
from metrics import find_ball_centroids, to_hwc_uint8
from models import PixelTransformer, SimpleJEPA, VelocityBaseline
from rollout import compute_metrics_for_rollout, rollout_jepa, rollout_transformer, rollout_velocity
from utils import ensure_dirs, get_device, load_checkpoint
from validate_disentanglement import run as run_disentanglement

OUT_DIR = "outputs"


def figure1_rollout_with_errorbars(num_steps=15, num_starts=5, seed=0, out_dir=OUT_DIR,
                                    frames_path="data/frames.npy", positions_path="data/positions.npy",
                                    ckpt_dir="checkpoints"):
    device = get_device()
    rng = np.random.default_rng(seed)

    frames = load_frames(frames_path)
    positions = load_positions(positions_path)

    jepa_path = os.path.join(ckpt_dir, "jepa_with_decoder.pth")
    if not os.path.exists(jepa_path):
        raise FileNotFoundError(
            f"{jepa_path} not found. Run train_jepa.py then train_decoder.py first."
        )
    jepa = SimpleJEPA(latent_dim=64).to(device)
    load_checkpoint(jepa, jepa_path, device=device)
    jepa.eval()

    trans = PixelTransformer().to(device)
    load_checkpoint(trans, os.path.join(ckpt_dir, "transformer_best.pth"), device=device)
    trans.eval()

    vel = VelocityBaseline().load(os.path.join(ckpt_dir, "velocity_baseline.pkl"))

    valid_range = (200, len(frames) - num_steps - 10)
    start_idxs = rng.integers(valid_range[0], valid_range[1], size=num_starts)

    all_runs = {"JEPA": [], "Pixel Transformer": [], "Velocity Baseline": []}
    for start_idx in start_idxs:
        start_idx = int(start_idx)
        gt_frames = frames[start_idx:start_idx + num_steps + 1]

        jepa_preds = rollout_jepa(jepa, gt_frames[0], num_steps, device)
        trans_preds = rollout_transformer(trans, gt_frames[0], num_steps, device)
        vel_preds = rollout_velocity(vel, gt_frames[0], gt_frames[1], num_steps)

        all_runs["JEPA"].append(compute_metrics_for_rollout(jepa_preds, gt_frames, positions, start_idx))
        all_runs["Pixel Transformer"].append(compute_metrics_for_rollout(trans_preds, gt_frames, positions, start_idx))
        all_runs["Velocity Baseline"].append(compute_metrics_for_rollout(vel_preds, gt_frames, positions, start_idx))

    colors = {"JEPA": "tab:blue", "Pixel Transformer": "tab:orange", "Velocity Baseline": "tab:green"}
    steps_x = np.arange(1, num_steps + 1)
    panels = [
        ("pos_err_mean", "Position Error (px)"),
        ("pixel_mse", "Pixel MSE (full image)"),
        ("foreground_mse", "Foreground MSE (ball bbox)"),
        ("ssim", "SSIM"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for (key, ylabel), ax in zip(panels, axes.flat):
        for name, runs in all_runs.items():
            arr = np.array([[np.nan if v is None else v for v in run[key]] for run in runs], dtype=np.float64)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            ax.plot(steps_x, mean, marker="o", label=name, color=colors[name], linewidth=2)
            ax.fill_between(steps_x, mean - std, mean + std, color=colors[name], alpha=0.15)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Open-Loop Rollout Comparison (mean ± std over {num_starts} random starts)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(out_dir, "figure1_rollout_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")


def figure3_intervention_trajectories(out_dir=OUT_DIR):
    normal_path = os.path.join(out_dir, "jepa_normal_frames.pt")
    interv_path = os.path.join(out_dir, "jepa_intervened_frames.pt")
    if not (os.path.exists(normal_path) and os.path.exists(interv_path)):
        print(f"  [skip] figure3 requires intervene.py outputs "
              f"({normal_path}, {interv_path}) — run intervene.py first.")
        return

    normal_frames = torch.load(normal_path)
    interv_frames = torch.load(interv_path)

    def trajectory(frames):
        red_xy, blue_xy = [], []
        for t in range(frames.shape[0]):
            c = find_ball_centroids(to_hwc_uint8(frames[t]))
            red_xy.append(c["red"])
            blue_xy.append(c["blue"])
        return red_xy, blue_xy

    normal_red, normal_blue = trajectory(normal_frames)
    interv_red, interv_blue = trajectory(interv_frames)

    intervene_step = 5  # matches intervene.py default INTERVENE_AT_STEP

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, red_xy, blue_xy, title in (
        (axes[0], normal_red, normal_blue, "Normal rollout"),
        (axes[1], interv_red, interv_blue, "Intervened rollout"),
    ):
        rx = [p[0] for p in red_xy if p is not None]
        ry = [p[1] for p in red_xy if p is not None]
        bx = [p[0] for p in blue_xy if p is not None]
        by = [p[1] for p in blue_xy if p is not None]
        ax.plot(rx, ry, "-o", color="crimson", label="Red ball", markersize=5)
        ax.plot(bx, by, "-o", color="steelblue", label="Blue ball", markersize=5)
        if len(rx) > intervene_step:
            ax.scatter([rx[intervene_step]], [ry[intervene_step]], color="black", marker="x",
                       s=120, zorder=5, label="Intervention point")
        ax.set_xlim(0, 64)
        ax.set_ylim(64, 0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    if len(normal_red) > intervene_step and normal_red[intervene_step] and interv_red[-1] and normal_red[-1]:
        pixel_shift = interv_red[-1][0] - normal_red[-1][0]
        caption = (f"Intervening on the identified X-position dimension at step {intervene_step} "
                   f"shifts the red ball ~{pixel_shift:+.1f} px by the end of the rollout, "
                   f"relative to the un-intervened trajectory.")
    else:
        caption = "Intervention shifts the red ball's trajectory relative to the normal rollout."

    fig.suptitle("JEPA Latent-Space Intervention: Ball Trajectories", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig_path = os.path.join(out_dir, "figure3_intervention_trajectories.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")


def run(out_dir=OUT_DIR, num_steps=15, num_starts=5, seed=0):
    ensure_dirs(out_dir)
    print("=== Figure 0: latent-space rollout error + linear probes (primary JEPA evaluation) ===")
    from evaluate_latent_dynamics import run as run_latent_eval
    run_latent_eval(out_dir=out_dir, num_steps=num_steps, num_starts=max(num_starts, 20), seed=seed)

    print("\n=== Figure 1: pixel-space rollout metrics (mean ± std over random starts) ===")
    figure1_rollout_with_errorbars(num_steps=num_steps, num_starts=num_starts, seed=seed, out_dir=out_dir)

    print("\n=== Figure 2: disentanglement heatmap ===")
    run_disentanglement(out_dir=out_dir)  # writes disentanglement_heatmap.png directly

    print("\n=== Figure 3: intervention trajectories ===")
    figure3_intervention_trajectories(out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--num-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(num_steps=args.num_steps, num_starts=args.num_starts, seed=args.seed)
