"""Vision feature extractors for reinforcement learning policies."""
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


def _prepare_image_channels(x: torch.Tensor, target_channels: int = 3) -> torch.Tensor:
    """Collapse stacked frames to a 3-channel tensor for pretrained backbones."""

    if x.dim() == 5:
        # Handle (n_envs, n_stack, c, h, w)
        x = x.flatten(0, 1)

    channels = x.shape[1]

    # If channels are a multiple of 3, interpret as stacked RGB frames
    if channels % 3 == 0 and channels > target_channels:
        n_frames = channels // 3
        x = x.view(x.shape[0], n_frames, 3, x.shape[2], x.shape[3])
        x = x.mean(dim=1)
    elif channels == 1:
        x = x.repeat(1, target_channels, 1, 1)
    elif channels != target_channels:
        # Generic fallback: average across channels then repeat
        x = x.mean(dim=1, keepdim=True).repeat(1, target_channels, 1, 1)

    return x


class MobileNetFeatureExtractor(BaseFeaturesExtractor):
    """Frozen MobileNetV3 feature extractor with a small projection head."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        image_space = observation_space["image"]
        n_input_channels = image_space.shape[0]

        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.features.requires_grad_(False)
        backbone.classifier = torch.nn.Identity()
        self.backbone = backbone

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_space.shape[1], image_space.shape[2])
            n_flatten = self.backbone(dummy).shape[1]

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.LayerNorm(features_dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        image = observations["image"].float() / 255.0
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = _prepare_image_channels(image, 3)
        feats = self.backbone(image)
        return self.projector(feats)
