"""
metrics.py — All evaluation functions: position error, pixel/foreground MSE, SSIM.
"""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim


def find_ball_centroids(img: np.ndarray):
    """
    img: (H, W, 3) uint8 or float [0,1] RGB image.
    Returns dict with 'red' and 'blue' centroids as (x, y) or None if not found.

    Uses simple color-channel thresholding + center-of-mass, matching how the
    scene is rendered (pure red / pure blue circles on a black background).
    """
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    r, g, b = img[..., 0].astype(np.int32), img[..., 1].astype(np.int32), img[..., 2].astype(np.int32)

    red_mask = (r > 60) & (r > g + 20) & (r > b + 20)
    blue_mask = (b > 60) & (b > r + 20) & (b > g + 20)

    def centroid(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    return {"red": centroid(red_mask), "blue": centroid(blue_mask)}


def position_error(pred_img: np.ndarray, gt_pos_red, gt_pos_blue):
    """
    L2 pixel distance between predicted (decoded) ball centroids and ground-truth
    physics-engine positions. Returns (red_err, blue_err, mean_err); a ball
    that can't be located in the prediction contributes NaN (excluded from mean).
    """
    centroids = find_ball_centroids(pred_img)
    errs = {}
    for name, gt in (("red", gt_pos_red), ("blue", gt_pos_blue)):
        c = centroids[name]
        errs[name] = float(np.hypot(c[0] - gt[0], c[1] - gt[1])) if c is not None else np.nan
    valid = [v for v in errs.values() if not np.isnan(v)]
    mean_err = float(np.mean(valid)) if valid else np.nan
    return errs["red"], errs["blue"], mean_err


def pixel_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Full-image MSE. pred, gt: (H, W, 3) in the same scale (float [0,1] recommended)."""
    return float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))


def foreground_mse(pred: np.ndarray, gt: np.ndarray, pad: int = 3) -> float:
    """
    MSE restricted to the bounding box around the balls in the ground-truth
    frame (a tight window around both centroids, plus `pad` px margin).
    Falls back to full-image MSE if no balls are detected in gt.
    """
    centroids = find_ball_centroids(gt)
    pts = [c for c in centroids.values() if c is not None]
    if not pts:
        return pixel_mse(pred, gt)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    h, w = gt.shape[:2]
    x0 = max(0, int(min(xs)) - 8 - pad)
    x1 = min(w, int(max(xs)) + 8 + pad)
    y0 = max(0, int(min(ys)) - 8 - pad)
    y1 = min(h, int(max(ys)) + 8 + pad)

    pred_c = pred[y0:y1, x0:x1].astype(np.float32)
    gt_c = gt[y0:y1, x0:x1].astype(np.float32)
    if pred_c.size == 0:
        return pixel_mse(pred, gt)
    return float(np.mean((pred_c - gt_c) ** 2))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Structural Similarity Index between predicted and ground-truth frames."""
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    data_range = 255.0 if pred_f.max() > 1.5 or gt_f.max() > 1.5 else 1.0
    return float(sk_ssim(gt_f, pred_f, data_range=data_range, channel_axis=-1))


def to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) torch tensor in [0,1] -> (H, W, C) uint8 numpy array."""
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def to_hwc_float(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) torch tensor in [0,1] -> (H, W, C) float32 numpy array in [0,1]."""
    return t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def mutual_information_binned(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """
    Simple histogram-based mutual information estimate between two 1D
    continuous variables, in nats. Used for the latent-dim <-> factor MI heatmap.
    """
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x * p_y)
        terms = p_xy * np.log(ratio)
    terms = np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    return float(terms.sum())
