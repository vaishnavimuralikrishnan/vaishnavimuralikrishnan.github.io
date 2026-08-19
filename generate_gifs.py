"""
generate_gifs.py — Build all publication-quality GIFs from rollout.py / intervene.py
outputs (256x256 upsampled, captioned, LANCZOS resampling).

Requires rollout.py and intervene.py to have been run first (they save the
underlying frame tensors as .pt files in outputs/).
"""
import argparse
import os

import torch

from data import load_frames
from utils import ensure_dirs, frames_to_gif

OUT_DIR = "outputs"


def run(out_dir=OUT_DIR, frames_path="data/frames.npy", start_idx=500):
    ensure_dirs(out_dir)

    def maybe_load(name):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return torch.load(path)
        print(f"  [skip] {path} not found — run rollout.py / intervene.py first.")
        return None

    gt_rollout = maybe_load("gt_rollout_frames.pt")
    jepa_rollout = maybe_load("jepa_rollout_frames.pt")
    trans_rollout = maybe_load("transformer_rollout_frames.pt")
    vel_rollout = maybe_load("velocity_rollout_frames.pt")
    jepa_intervened = maybe_load("jepa_intervened_frames.pt")

    if gt_rollout is not None:
        # Ground truth sample: first 20 raw frames from the test-set region.
        frames = load_frames(frames_path)
        gt_sample = frames[start_idx:start_idx + 20]
        frames_to_gif(gt_sample, os.path.join(out_dir, "gt_sample.gif"), model_name="Ground Truth",
                       start_frame_num=start_idx)

    if jepa_rollout is not None:
        frames_to_gif(jepa_rollout, os.path.join(out_dir, "jepa_rollout.gif"), model_name="JEPA")

    if trans_rollout is not None:
        frames_to_gif(trans_rollout, os.path.join(out_dir, "transformer_rollout.gif"), model_name="Pixel Transformer")

    if vel_rollout is not None:
        frames_to_gif(vel_rollout, os.path.join(out_dir, "velocity_rollout.gif"), model_name="Velocity Baseline")

    if jepa_intervened is not None:
        frames_to_gif(jepa_intervened, os.path.join(out_dir, "jepa_intervention.gif"), model_name="JEPA (intervened)")

    print("\nAll available GIFs generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=500)
    args = parser.parse_args()
    run(start_idx=args.start_idx)
