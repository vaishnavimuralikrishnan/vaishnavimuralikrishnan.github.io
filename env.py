"""
env.py — Physics simulation and data generation for the Bouncing Balls dataset.

Simulates two elastically-colliding balls (red: mass=2, radius=6; blue: mass=1,
radius=4) inside a 64x64 pixel arena using pymunk, and renders each frame to a
clean, antialiasing-free RGB image. Frames are saved as uint8 to data/frames.npy.

Ground-truth ball positions (pixel coordinates) are also saved to
data/positions.npy with shape (N, 2, 2) -> [frame, ball(red=0/blue=1), (x, y)].
This is required later for position-error metrics and MI/disentanglement
analysis, since re-deriving "ground truth" from rendered pixels alone is lossy.

Usage:
    python env.py [--num-frames 10000] [--seed 0]
"""
import argparse
import os

import numpy as np
import pymunk
from PIL import Image, ImageDraw

WIDTH = 64
HEIGHT = 64
DEFAULT_NUM_FRAMES = 10000
OUTPUT_FRAMES_PATH = "data/frames.npy"
OUTPUT_POSITIONS_PATH = "data/positions.npy"

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def build_space(seed: int = 0) -> tuple[pymunk.Space, pymunk.Body, pymunk.Body]:
    """Construct the pymunk space, static walls, and the two ball bodies."""
    rng = np.random.default_rng(seed)

    space = pymunk.Space()
    space.gravity = (0, 0)  # pure elastic bouncing, no gravity

    wall_defs = [
        ((0, 0), (0, HEIGHT)),      # left
        ((WIDTH, 0), (WIDTH, HEIGHT)),  # right
        ((0, HEIGHT), (WIDTH, HEIGHT)),  # bottom
        ((0, 0), (WIDTH, 0)),       # top
    ]
    for p1, p2 in wall_defs:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, p1, p2, 1)
        # CRITICAL: walls must be elastic too, or ball-wall collisions damp
        # out over time (pymunk defaults shape.elasticity to 0.0).
        shape.elasticity = 1.0
        shape.friction = 0.0
        space.add(body, shape)

    # Ball 1: Red, larger, starting left-ish
    mass1, radius1 = 2, 6
    body1 = pymunk.Body(mass1, pymunk.moment_for_circle(mass1, 0, radius1))
    body1.position = (16, 32)
    body1.velocity = (55, 37)
    shape1 = pymunk.Circle(body1, radius1)
    shape1.elasticity = 1.0
    shape1.friction = 0.0
    shape1.color = (*RED, 255)
    space.add(body1, shape1)

    # Ball 2: Blue, smaller, starting right-ish
    mass2, radius2 = 1, 4
    body2 = pymunk.Body(mass2, pymunk.moment_for_circle(mass2, 0, radius2))
    body2.position = (48, 32)
    body2.velocity = (-43, -29)
    shape2 = pymunk.Circle(body2, radius2)
    shape2.elasticity = 1.0
    shape2.friction = 0.0
    shape2.color = (*BLUE, 255)
    space.add(body2, shape2)

    return space, body1, body2


def render_frame(space: pymunk.Space, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Render the current physics state to an (H, W, 3) uint8 array, no AA."""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    for shape in space.shapes:
        if isinstance(shape, pymunk.Circle):
            pos = shape.body.position
            r = shape.radius
            color = tuple(shape.color[:3])
            x, y = pos.x, pos.y
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)
    return np.array(img, dtype=np.uint8)


def generate_dataset(num_frames: int = DEFAULT_NUM_FRAMES, seed: int = 0,
                      sim_steps_per_frame: int = 10, dt: float = 0.01):
    space, red_body, blue_body = build_space(seed=seed)

    frames = np.empty((num_frames, HEIGHT, WIDTH, 3), dtype=np.uint8)
    positions = np.empty((num_frames, 2, 2), dtype=np.float32)  # [frame, ball, xy]

    print(f"Generating {num_frames} frames (seed={seed})...")
    for i in range(num_frames):
        for _ in range(sim_steps_per_frame):
            space.step(dt)

        frames[i] = render_frame(space)
        positions[i, 0] = (red_body.position.x, red_body.position.y)
        positions[i, 1] = (blue_body.position.x, blue_body.position.y)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{num_frames} frames "
                  f"(red speed={red_body.velocity.length:.1f}, "
                  f"blue speed={blue_body.velocity.length:.1f})")

    return frames, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames, positions = generate_dataset(num_frames=args.num_frames, seed=args.seed)

    os.makedirs(os.path.dirname(OUTPUT_FRAMES_PATH), exist_ok=True)
    np.save(OUTPUT_FRAMES_PATH, frames)
    np.save(OUTPUT_POSITIONS_PATH, positions)

    print(f"Saved frames {frames.shape} -> {OUTPUT_FRAMES_PATH}")
    print(f"Saved positions {positions.shape} -> {OUTPUT_POSITIONS_PATH}")
    print(f"Mean abs pixel value (sanity, should be > a few and not decaying): "
          f"{frames.astype(np.float32).mean():.3f}")


if __name__ == "__main__":
    main()
