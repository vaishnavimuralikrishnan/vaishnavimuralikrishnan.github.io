"""
find_position_dimension.py — Sweep every JEPA latent dimension and measure how
much perturbing it shifts each ball's decoded position. Identifies which
dimensions encode X/Y position for red/blue.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import load_frames
from metrics import find_ball_centroids, to_hwc_uint8
from models import SimpleJEPA
from utils import ensure_dirs, get_device, load_checkpoint

LATENT_DIM = 64
PERTURBATION = 5.0


def run(frame_idx=500, ckpt_path="checkpoints/jepa_with_decoder.pth", frames_path="data/frames.npy",
        out_dir="outputs", perturbation=PERTURBATION):
    ensure_dirs(out_dir)
    device = get_device()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. This script needs a trained decoder — "
            f"run train_jepa.py then train_decoder.py first."
        )
    frames = load_frames(frames_path)
    model = SimpleJEPA(latent_dim=LATENT_DIM).to(device)
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    x = frames[frame_idx].unsqueeze(0).to(device)
    with torch.no_grad():
        z_base = model.encode(x)
        base_img = to_hwc_uint8(model.decode(z_base).squeeze(0))

    base_centroids = find_ball_centroids(base_img)
    print(f"Frame {frame_idx} base centroids: {base_centroids}")
    if base_centroids["red"] is None or base_centroids["blue"] is None:
        print("WARNING: one or both balls not detected in the base decoded image; "
              "try a different --frame-idx.")

    rows = []  # dim, dx_red, dy_red, dx_blue, dy_blue, max_abs_shift
    with torch.no_grad():
        for dim in range(LATENT_DIM):
            z_mod = z_base.clone()
            z_mod[0, dim] += perturbation
            mod_img = to_hwc_uint8(model.decode(z_mod).squeeze(0))
            mod_centroids = find_ball_centroids(mod_img)

            def delta(name, axis):
                b, m = base_centroids[name], mod_centroids[name]
                if b is None or m is None:
                    return 0.0
                return m[axis] - b[axis]

            dx_red, dy_red = delta("red", 0), delta("red", 1)
            dx_blue, dy_blue = delta("blue", 0), delta("blue", 1)
            max_shift = max(abs(dx_red), abs(dy_red), abs(dx_blue), abs(dy_blue))
            rows.append((dim, dx_red, dy_red, dx_blue, dy_blue, max_shift))

    csv_path = os.path.join(out_dir, "dimension_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "dx_red", "dy_red", "dx_blue", "dy_blue", "max_abs_shift"])
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    x_scores = [(r[0], abs(r[1]) + abs(r[3])) for r in rows]  # combined |dx_red|+|dx_blue|
    y_scores = [(r[0], abs(r[2]) + abs(r[4])) for r in rows]
    top_x = sorted(x_scores, key=lambda t: t[1], reverse=True)[:5]
    top_y = sorted(y_scores, key=lambda t: t[1], reverse=True)[:5]

    print("\n=== Top 5 X-control dimensions ===")
    for dim, score in top_x:
        print(f"  dim {dim}: combined |dx| = {score:.3f}")
    print("\n=== Top 5 Y-control dimensions ===")
    for dim, score in top_y:
        print(f"  dim {dim}: combined |dy| = {score:.3f}")

    # Summary figure: bar chart of max_abs_shift per dimension
    dims = [r[0] for r in rows]
    shifts = [r[5] for r in rows]
    top_x_dims = {d for d, _ in top_x}
    top_y_dims = {d for d, _ in top_y}
    colors = []
    for d in dims:
        if d in top_x_dims and d in top_y_dims:
            colors.append("purple")
        elif d in top_x_dims:
            colors.append("crimson")
        elif d in top_y_dims:
            colors.append("steelblue")
        else:
            colors.append("lightgray")

    plt.figure(figsize=(14, 5))
    plt.bar(dims, shifts, color=colors)
    plt.xlabel("Latent dimension")
    plt.ylabel("Max |position shift| (pixels)")
    plt.title(f"JEPA latent-dimension sensitivity (perturbation=±{perturbation}, frame {frame_idx})")
    plt.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="crimson", label="Top-5 X-control"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Top-5 Y-control"),
        plt.Rectangle((0, 0), 1, 1, color="purple", label="Both"),
    ]
    plt.legend(handles=handles)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "dimension_sensitivity.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")

    return top_x, top_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-idx", type=int, default=500)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION)
    args = parser.parse_args()
    run(frame_idx=args.frame_idx, perturbation=args.perturbation)
