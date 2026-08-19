"""
intervene.py — Run a 10-step JEPA rollout, intervening on the identified
X-position latent dimension at step 5, and quantify the resulting directional
shift in the decoded ball position.
"""
import argparse
import csv
import os

import torch

from data import load_frames
from metrics import find_ball_centroids, to_hwc_uint8
from models import SimpleJEPA
from utils import ensure_dirs, frames_to_gif, get_device, load_checkpoint

NUM_STEPS = 10
INTERVENE_AT_STEP = 5
SHIFT_VALUE = 3.0


def load_top_x_dim(out_dir="outputs"):
    """Read the X-control dimension identified by find_position_dimension.py."""
    csv_path = os.path.join(out_dir, "dimension_sensitivity.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found — run find_position_dimension.py first."
        )
    best_dim, best_score = None, -1.0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = abs(float(row["dx_red"])) + abs(float(row["dx_blue"]))
            if score > best_score:
                best_score = score
                best_dim = int(row["dim"])
    return best_dim


def run(start_idx=500, num_steps=NUM_STEPS, intervene_at=INTERVENE_AT_STEP,
        shift_value=SHIFT_VALUE, ckpt_path="checkpoints/jepa_with_decoder.pth",
        frames_path="data/frames.npy", out_dir="outputs", x_dim=None):
    ensure_dirs(out_dir)
    device = get_device()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Intervention visualization needs a trained decoder — "
            f"run train_jepa.py then train_decoder.py first."
        )

    if x_dim is None:
        x_dim = load_top_x_dim(out_dir)
    print(f"Intervening on latent dimension {x_dim} (X-position control) by +{shift_value}")

    frames = load_frames(frames_path)
    ctx = frames[start_idx].unsqueeze(0).to(device)

    model = SimpleJEPA(latent_dim=64).to(device)
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    normal_imgs, interv_imgs = [ctx.squeeze(0).cpu()], [ctx.squeeze(0).cpu()]
    normal_centroids, interv_centroids = [], []

    with torch.no_grad():
        z_normal = model.encode(ctx)
        z_interv = z_normal.clone()

        for step in range(num_steps):
            z_normal = model.predict(z_normal)
            img_normal = model.decode(z_normal)
            normal_imgs.append(img_normal.squeeze(0).cpu())
            normal_centroids.append(find_ball_centroids(to_hwc_uint8(img_normal.squeeze(0))))

            z_interv = model.predict(z_interv)
            if step == intervene_at - 1:  # intervene right after producing step `intervene_at`
                print(f"  -> perturbing dim {x_dim} at step {step + 1}")
                z_interv = z_interv.clone()
                z_interv[:, x_dim] += shift_value
            img_interv = model.decode(z_interv)
            interv_imgs.append(img_interv.squeeze(0).cpu())
            interv_centroids.append(find_ball_centroids(to_hwc_uint8(img_interv.squeeze(0))))

    normal_stack = torch.stack(normal_imgs, dim=0)
    interv_stack = torch.stack(interv_imgs, dim=0)

    frames_to_gif(normal_stack, os.path.join(out_dir, "jepa_rollout_normal.gif"), model_name="JEPA (normal)")
    frames_to_gif(interv_stack, os.path.join(out_dir, "jepa_intervention.gif"), model_name="JEPA (intervened)")

    # Quantify the shift: compare red-ball x-centroid right before vs. several
    # steps after the intervention.
    pre_c = normal_centroids[intervene_at - 1]["red"]
    post_c_normal = normal_centroids[-1]["red"]
    post_c_interv = interv_centroids[-1]["red"]

    if pre_c and post_c_normal and post_c_interv:
        shift_normal = post_c_normal[0] - pre_c[0]
        shift_interv = post_c_interv[0] - pre_c[0]
        delta_from_intervention = post_c_interv[0] - post_c_normal[0]
        direction = "rightward (+x)" if delta_from_intervention > 0 else "leftward (-x)"
        print(f"\nRed ball X position at step {intervene_at}: {pre_c[0]:.2f}")
        print(f"Final X (normal rollout):      {post_c_normal[0]:.2f} (Δ={shift_normal:+.2f})")
        print(f"Final X (intervened rollout):  {post_c_interv[0]:.2f} (Δ={shift_interv:+.2f})")
        print(f"Net effect of intervention:    {delta_from_intervention:+.2f} px, {direction}")
        expected = "rightward" if shift_value > 0 else "leftward"
        matches = expected in direction
        print(f"Expected direction: {expected} -> {'MATCHES' if matches else 'DOES NOT MATCH'}")
    else:
        print("WARNING: could not locate red ball centroid at all required steps; "
              "try a different --start-idx or --x-dim.")
        delta_from_intervention = float("nan")

    summary_path = os.path.join(out_dir, "intervention_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_dim", "shift_value", "intervene_at_step",
                          "pre_x", "post_x_normal", "post_x_intervened", "net_effect_px"])
        writer.writerow([x_dim, shift_value, intervene_at,
                          pre_c[0] if pre_c else "", post_c_normal[0] if post_c_normal else "",
                          post_c_interv[0] if post_c_interv else "", delta_from_intervention])
    print(f"Saved {summary_path}")

    torch.save(normal_stack, os.path.join(out_dir, "jepa_normal_frames.pt"))
    torch.save(interv_stack, os.path.join(out_dir, "jepa_intervened_frames.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--intervene-at", type=int, default=INTERVENE_AT_STEP)
    parser.add_argument("--shift-value", type=float, default=SHIFT_VALUE)
    parser.add_argument("--x-dim", type=int, default=None, help="Override auto-detected X-control dim")
    args = parser.parse_args()
    run(start_idx=args.start_idx, num_steps=args.num_steps, intervene_at=args.intervene_at,
        shift_value=args.shift_value, x_dim=args.x_dim)
