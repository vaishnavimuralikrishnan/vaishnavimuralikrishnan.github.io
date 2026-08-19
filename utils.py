"""
utils.py — Visualization helpers, checkpoint I/O, and misc shared utilities.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from metrics import to_hwc_uint8

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _load_font(size: int = 12):
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


def frames_to_gif(frames: torch.Tensor, filename: str, model_name: str = "",
                   upscale: int = 4, duration_ms: int = 200, start_frame_num: int = 0):
    """
    frames: (T, C, H, W) torch tensor in [0, 1].
    Saves a publication-quality GIF, upsampled with LANCZOS resampling, with an
    overlaid "frame N | model_name" caption on every frame.
    """
    font = _load_font(11)
    pil_frames = []
    for i in range(frames.shape[0]):
        img_np = to_hwc_uint8(frames[i])
        img = Image.fromarray(img_np, mode="RGB")
        img = img.resize((img.width * upscale, img.height * upscale), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(img)
        caption = f"frame {start_frame_num + i} | {model_name}" if model_name else f"frame {start_frame_num + i}"
        # Small dark backing rectangle for legibility over any background.
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        pad = 3
        draw.rectangle(
            [2, 2, text_bbox[2] - text_bbox[0] + 2 * pad + 2, text_bbox[3] - text_bbox[1] + 2 * pad + 2],
            fill=(0, 0, 0),
        )
        draw.text((2 + pad, 2 + pad), caption, fill=(255, 255, 255), font=font)
        pil_frames.append(img)

    pil_frames[0].save(
        filename, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    print(f"Saved GIF: {filename}")


def save_checkpoint(model: torch.nn.Module, path: str):
    ensure_dirs(os.path.dirname(path) or ".")
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
