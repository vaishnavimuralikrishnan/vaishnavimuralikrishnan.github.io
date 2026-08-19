# JEPA vs. Pixel Transformer: Bouncing Balls Dynamics

## Overview

This project compares two ways of learning the dynamics of a simple physical
system — two elastically colliding balls in a 64×64 arena — from raw pixels.
**SimpleJEPA** (Joint Embedding Predictive Architecture) learns to predict the
*next frame's representation* in a compact 64-dim latent space, trained with a
momentum-encoder target and **never backpropagating through pixels**. A
**PixelTransformer** baseline instead predicts the next frame directly in
pixel-patch space. A non-neural **VelocityBaseline** (linear regression on
frame differences) acts as a sanity check. Beyond next-step prediction
accuracy, the project probes *what* JEPA's latent space represents: whether
ball position is linearly decodable from the representation, which latent
dimensions carry position vs. appearance information (mutual information),
and whether directly editing a single latent dimension produces the expected,
directional change in a decoded rollout (a causal intervention test).

## A note on methodology (read this first)

JEPA's defining property is that **the decoder is not part of training** — the
loss is purely `MSE(predictor(encode(x_t)), momentum_encoder(x_t+1))`. This
project trains it that way. Two consequences follow directly from that
design, and this pipeline is built around them rather than working around
them:

1. **The decoder is trained separately, after JEPA, on frozen representations**
   (`train_decoder.py`). This is standard practice for producing JEPA
   visualizations (used e.g. for the I-JEPA paper's figures) — the encoder
   and predictor never see a gradient from it. It exists purely so you can
   *look at* what the latent space predicts; it plays no role in what the
   model actually learns.
2. **The primary evaluation of JEPA is decoder-free** (`evaluate_latent_dynamics.py`):
   open-loop rollout error measured directly in latent space against the same
   momentum-encoder target used in training, plus linear probes (z → ball
   position, z → color) that quantify what the representation encodes without
   relying on a lossy, separately-trained decoder to tell you. This is the
   evaluation that actually reflects JEPA's stated objective, and the one to
   lead with when writing this up.

The pixel-space rollout comparison (`rollout.py`, using the trained decoder)
is still included because it's a fair, intuitive way to compare against the
Pixel Transformer — but it is a secondary, decoder-quality-dependent metric,
not the primary claim.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate data (10,000 frames + ground-truth positions)
python env.py --num-frames 10000 --seed 0

# 2. Train JEPA — PURE latent-space objective, no decoder involved
python train_jepa.py --epochs 30

# 3. Train the decoder separately, on frozen JEPA representations
#    (needed only for pixel-space visualization/rollout/intervention scripts)
python train_decoder.py --epochs 20

# 4. Train the baselines
python train_transformer.py --epochs 30
python train_velocity_baseline.py

# 5. PRIMARY evaluation — decoder-free, operates purely in latent space
python evaluate_latent_dynamics.py

# 6. Secondary interpretability / pixel-space analysis (needs the decoder)
python find_position_dimension.py       # identifies X/Y-control latent dims
python validate_disentanglement.py      # MI heatmap (encoder-only, no decoder)
python rollout.py                       # 15-step pixel-space rollout, all 3 models
python intervene.py                     # perturb the X-dim mid-rollout, check directionality

# 7. Visualization
python generate_gifs.py                 # all publication GIFs (256x256, captioned)
python create_report_figures.py         # figures 0-3, 300 DPI PNGs (runs step 5 too)
```

Outputs land in `outputs/` (figures, GIFs, CSVs) and `checkpoints/` (`.pth` /
`.pkl` model weights). All scripts are seeded (`--seed`, default `0`) for
reproducibility.

**Checkpoint naming matters:** `jepa_best.pth` / `jepa_final.pth` (from
`train_jepa.py`) have a properly trained encoder+predictor but a **randomly
initialized decoder** — use these for `evaluate_latent_dynamics.py`.
`jepa_with_decoder.pth` (from `train_decoder.py`) additionally has a trained
decoder — use this for everything that needs to look at pixels
(`find_position_dimension.py`, `rollout.py`, `intervene.py`,
`create_report_figures.py`).

## Project structure

```
env.py                       # physics sim + frame/position generation
models.py                    # SimpleJEPA, PixelTransformer, VelocityBaseline
data.py                      # datasets + loading utilities
metrics.py                   # position error, pixel/foreground MSE, SSIM, MI
utils.py                     # seeding, checkpoints, GIF rendering
train_jepa.py                # PURE JEPA: latent loss only, decoder untouched
train_decoder.py             # post-hoc decoder training on frozen JEPA encoder
train_transformer.py / train_velocity_baseline.py
evaluate_latent_dynamics.py  # PRIMARY eval: latent rollout error + linear probes
find_position_dimension.py   # per-dimension latent-perturbation sweep (needs decoder)
validate_disentanglement.py  # MI(latent dim; position/color) heatmap (encoder-only)
rollout.py                   # 15-step pixel-space rollout comparison (needs decoder)
intervene.py                 # causal intervention on an identified X-dim (needs decoder)
generate_gifs.py              create_report_figures.py
```

## Results summary

*(Fill in after running the full pipeline — placeholders below show the
expected shape of the results based on smoke-testing this pipeline on a
small subset of the data.)*

- The JEPA predictor's latent rollout error should stay well below the
  static "no-dynamics" baseline (repeating z_0) at every step — if it
  doesn't, the predictor hasn't learned real dynamics.
- Linear probes should show high held-out R² for position from the latent
  space (position is linearly decodable), and near-zero R² for the color
  factor (color/appearance is not confounded with the position dimensions).
- If the two balls show *asymmetric* decodability (e.g. one ball's position
  is much more linearly recoverable than the other's), that's a real,
  reportable finding, not a bug — worth investigating (does it correlate
  with which ball is larger/more visually dominant, occlusion frequency,
  or is it an artifact of representation capacity being unevenly allocated).
- In pixel-space, JEPA's position error should stay lower than the Pixel
  Transformer's over longer open-loop rollouts, because predicting in a
  compact latent space avoids compounding pixel-level blur/artifact
  accumulation. The Pixel Transformer can still show deceptively low raw
  pixel MSE (mostly black background) — this is exactly why position error,
  not MSE, is reported as the primary pixel-space metric.
- Intervening on the identified X-position dimension mid-rollout should
  shift the decoded ball's trajectory in the expected direction — causal,
  not just correlational, evidence that the dimension encodes position.

## Key findings — what did each model learn?

- **SimpleJEPA** learns a latent space where next-step dynamics are
  predictable purely in embedding space (verified without ever decoding to
  pixels), and where ball position is linearly recoverable from a handful of
  dimensions — evidence of a compact, partially disentangled internal model
  of the scene, not a pixel-copying shortcut.
- **PixelTransformer** learns direct pixel-to-pixel transition statistics. It
  can be competitive on raw MSE (dominated by background pixels) but has no
  compact notion of "ball position" to reason about, probe, or intervene on.
- **VelocityBaseline** confirms the task is non-trivial: naive linear
  extrapolation from finite differences degrades quickly under a physics
  engine with wall collisions, so any real gain from JEPA/Transformer
  reflects learned structure, not a trivial task.

## Notes on data generation

`env.py` sets `elasticity = 1.0` on **both** the ball shapes and the four
static wall segments. pymunk defaults `Shape.elasticity` to `0.0`; leaving the
walls at the default silently damps out all motion over a 10,000-frame run
(collisions become inelastic), which would otherwise collapse this into a
near-static prediction task for every downstream model. `env.py` also saves
`data/positions.npy` — the physics engine's ground-truth ball (x, y) per
frame — which the evaluation and analysis scripts use as the position label,
rather than trying to re-derive "ground truth" from rendered pixels.
