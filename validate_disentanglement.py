"""
validate_disentanglement.py — Quantify how disentangled the JEPA latent space is
by computing mutual information (MI) between each of the 64 latent dimensions
and three semantic factors: ball X position, ball Y position, and ball color.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_frames, load_positions
from metrics import mutual_information_binned
from models import SimpleJEPA
from utils import ensure_dirs, get_device, load_checkpoint

LATENT_DIM = 64
FACTOR_NAMES = ["pos_x", "pos_y", "color"]


def run(ckpt_path="checkpoints/jepa_best.pth", frames_path="data/frames.npy",
        positions_path="data/positions.npy", out_dir="outputs", num_samples=2000, seed=0):
    ensure_dirs(out_dir)
    device = get_device()
    rng = np.random.default_rng(seed)

    frames = load_frames(frames_path)
    positions = load_positions(positions_path)  # (N, 2, 2) -> [frame, ball(red/blue), xy]
    n = min(num_samples, len(frames))
    idx = rng.choice(len(frames), size=n, replace=False)
    idx.sort()

    model = SimpleJEPA(latent_dim=LATENT_DIM).to(device)
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    batch = frames[idx].to(device)
    with torch.no_grad():
        z = []
        bs = 256
        for i in range(0, len(batch), bs):
            z.append(model.encode(batch[i:i + bs]).cpu())
        z = torch.cat(z, dim=0).numpy()  # (n, 64)

    # Semantic factors: use the red ball's (x, y) as position ground truth (the
    # more visually dominant ball), and a scalar "color" factor = which ball is
    # closer to top-left vs bottom-right in RG-space is arbitrary here — instead
    # we use the RED/BLUE identity split as the color factor: for a 2-ball scene
    # we treat the red ball's channel dominance in the frame as a fixed constant,
    # so instead the "color" factor is the mean red-channel intensity of the
    # frame vs mean blue-channel intensity (captures color-appearance variation
    # driven by rendering/occlusion overlap between the two balls).
    pos_x = positions[idx, 0, 0]  # red ball x
    pos_y = positions[idx, 0, 1]  # red ball y
    frame_np = batch.cpu().numpy()  # (n, 3, 64, 64)
    color_factor = frame_np[:, 0].mean(axis=(1, 2)) - frame_np[:, 2].mean(axis=(1, 2))  # mean R - mean B

    factors = {"pos_x": pos_x, "pos_y": pos_y, "color": color_factor}

    mi_matrix = np.zeros((LATENT_DIM, 3))
    for d in range(LATENT_DIM):
        for j, fname in enumerate(FACTOR_NAMES):
            mi_matrix[d, j] = mutual_information_binned(z[:, d], factors[fname], bins=16)

    csv_path = os.path.join(out_dir, "mi_matrix.csv")
    np.savetxt(csv_path, mi_matrix, delimiter=",", header=",".join(FACTOR_NAMES), comments="")
    print(f"Saved {csv_path}")

    for j, fname in enumerate(FACTOR_NAMES):
        top_dims = np.argsort(mi_matrix[:, j])[::-1][:5]
        print(f"Top-5 dims by MI with {fname}: "
              + ", ".join(f"dim {d} (MI={mi_matrix[d, j]:.3f})" for d in top_dims))

    # Heatmap (log scale color intensity)
    plt.figure(figsize=(5, 12))
    log_mi = np.log1p(mi_matrix)
    im = plt.imshow(log_mi, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="log(1 + MI)")
    plt.xticks(range(3), FACTOR_NAMES)
    plt.yticks(range(LATENT_DIM), range(LATENT_DIM), fontsize=6)
    plt.ylabel("Latent dimension")
    plt.title("JEPA Latent Disentanglement\n(Mutual Information, log scale)")

    for j in range(3):
        top_dim = int(np.argmax(mi_matrix[:, j]))
        plt.gca().add_patch(
            plt.Rectangle((j - 0.5, top_dim - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
        )

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "disentanglement_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")

    return mi_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(num_samples=args.num_samples, seed=args.seed)
