"""
data.py — Dataset + data loading utilities shared by all training/analysis scripts.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from models import patchify

FRAMES_PATH = "data/frames.npy"
POSITIONS_PATH = "data/positions.npy"


def load_frames(path: str = FRAMES_PATH) -> torch.Tensor:
    """Load frames.npy (N, H, W, C) uint8 -> normalized (N, C, H, W) float32 tensor in [0, 1]."""
    frames = np.load(path).astype(np.float32) / 255.0
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


def load_positions(path: str = POSITIONS_PATH) -> np.ndarray:
    """Load ground-truth ball positions (N, 2, 2) -> [frame, ball(red=0/blue=1), (x, y)]."""
    return np.load(path)


def train_val_split(frames: torch.Tensor, val_fraction: float = 0.2):
    split_idx = int((1 - val_fraction) * len(frames))
    return frames[:split_idx], frames[split_idx:]


class FramePairDataset(Dataset):
    """Consecutive frame pairs (x_t, x_t+1) for JEPA / Transformer training."""

    def __init__(self, frames: torch.Tensor):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames) - 1

    def __getitem__(self, idx: int):
        return self.frames[idx], self.frames[idx + 1]


class FramePatchPairDataset(Dataset):
    """Consecutive frame pairs, pre-patchified for the PixelTransformer."""

    def __init__(self, frames: torch.Tensor):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames) - 1

    def __getitem__(self, idx: int):
        x = patchify(self.frames[idx:idx + 1]).squeeze(0)
        y = patchify(self.frames[idx + 1:idx + 2]).squeeze(0)
        return x, y


class SingleFrameDataset(Dataset):
    """Individual frames (no pairing), used for post-hoc decoder training."""

    def __init__(self, frames: torch.Tensor):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        return self.frames[idx]


class FrameTripletDataset(Dataset):
    """Consecutive frame triples (x_t, x_t+1, x_t+2) for the velocity baseline."""

    def __init__(self, frames: torch.Tensor):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames) - 2

    def __getitem__(self, idx: int):
        return self.frames[idx], self.frames[idx + 1], self.frames[idx + 2]


def flatten_frames(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) -> (N, C*H*W), used by the linear VelocityBaseline."""
    return x.reshape(x.shape[0], -1)


def unflatten_frames(x: torch.Tensor, shape=(3, 64, 64)) -> torch.Tensor:
    return x.reshape(x.shape[0], *shape)
