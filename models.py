"""
models.py — All model definitions: SimpleJEPA, PixelTransformer, VelocityBaseline.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# SimpleJEPA
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    """Conv4 encoder: 32 -> 64 -> 128 -> 256 channels, then linear -> latent_dim."""

    def __init__(self, latent_dim: int = 64, input_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),               # 32->16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),              # 16->8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True),             # 8->4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of Encoder. Used only for visualization/interpretability, never in the JEPA loss."""

    def __init__(self, latent_dim: int = 64, output_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4), nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),    # 16->32
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),              # 32->64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Predictor(nn.Module):
    """Latent-space-only predictor: 64 -> 128 -> 64."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture.

    - encoder / momentum_encoder share architecture, separate weights.
    - momentum_encoder is EMA-updated from encoder (never receives gradients).
    - predictor operates purely in latent space.
    - decoder exists ONLY for visualization/interpretability; it is not part
      of the training loss and receives no gradient from the JEPA objective.
    """

    def __init__(self, latent_dim: int = 64, input_channels: int = 3, momentum_tau: float = 0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.momentum_tau = momentum_tau

        self.encoder = Encoder(latent_dim, input_channels)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        for p in self.momentum_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    # ---- individual forward passes -------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_momentum(self, x: torch.Tensor) -> torch.Tensor:
        return self.momentum_encoder(x)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # ---- EMA update for momentum encoder --------------------------------
    @torch.no_grad()
    def update_momentum_encoder(self):
        tau = self.momentum_tau
        for p_online, p_target in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    # ---- training forward -----------------------------------------------
    def forward(self, x_t: torch.Tensor, x_t1: torch.Tensor):
        """
        x_t, x_t1: consecutive frames, (B, C, H, W), in [0, 1].
        Returns: loss, z_t, z_next_pred
        """
        z_t = self.encode(x_t)
        z_next_pred = self.predict(z_t)
        z_t1_target = self.encode_momentum(x_t1)  # no_grad internally
        loss = nn.functional.mse_loss(z_next_pred, z_t1_target)
        return loss, z_t, z_next_pred


# --------------------------------------------------------------------------
# PixelTransformer
# --------------------------------------------------------------------------
PATCH_SIZE = 8
NUM_PATCHES = (64 // PATCH_SIZE) ** 2  # 64
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # 192


def patchify(images: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(N, C, H, W) -> (N, num_patches, patch_dim)."""
    N, C, H, W = images.shape
    ph, pw = H // patch_size, W // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(N, C, ph, pw, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(N, ph * pw, C * patch_size * patch_size)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int = PATCH_SIZE,
               img_shape: tuple[int, int, int] = (3, 64, 64)) -> torch.Tensor:
    """(N, num_patches, patch_dim) -> (N, C, H, W)."""
    N, num_patches, _ = patches.shape
    C, H, W = img_shape
    ph, pw = H // patch_size, W // patch_size
    x = patches.view(N, ph, pw, C, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(N, C, H, W)
    return x


class PixelTransformer(nn.Module):
    """Direct pixel-patch next-frame predictor. No latent bottleneck."""

    def __init__(self, num_patches: int = NUM_PATCHES, embed_dim: int = PATCH_DIM,
                 num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, 192) patchified current frame -> (B, 64, 192) predicted next frame."""
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.output_proj(x)


# --------------------------------------------------------------------------
# VelocityBaseline
# --------------------------------------------------------------------------
class VelocityBaseline:
    """
    Non-neural sanity-check baseline.

    Learns x_{t+2} = W @ concat(x_t, velocity_t) + b via linear regression,
    where velocity_t = x_{t+1} - x_t. Operates on flattened, normalized pixels.
    """

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.fitted = False

    @staticmethod
    def _make_features(x_t, x_t1):
        """x_t, x_t1: (N, D) flattened frames -> features (N, 2D)."""
        velocity = x_t1 - x_t
        return torch.cat([x_t, velocity], dim=-1)

    def fit(self, x_t, x_t1, x_t2):
        """x_t, x_t1, x_t2: (N, D) flattened, normalized frames."""
        feats = self._make_features(x_t, x_t1).numpy()
        target = x_t2.numpy()
        self.model.fit(feats, target)
        self.fitted = True

    def predict(self, x_t, x_t1):
        feats = self._make_features(x_t, x_t1).numpy()
        pred = self.model.predict(feats)
        return torch.from_numpy(pred).float().clamp(0.0, 1.0)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.fitted = True
        return self
