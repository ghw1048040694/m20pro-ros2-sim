"""Shared compact M20 multimodal action-chunk model."""

from __future__ import annotations

import torch
from torch import nn


class M20VLAActionChunk(nn.Module):
    def __init__(self, horizon: int, architecture: str = "global_v1", stop_head: bool = False):
        super().__init__()
        self.horizon = horizon
        self.architecture = architecture
        self.stop_head_enabled = stop_head
        if architecture not in {"global_v1", "spatial_v2"}:
            raise ValueError(f"Unsupported M20 VLA architecture: {architecture}")
        image_pool = nn.AdaptiveAvgPool2d(1) if architecture == "global_v1" else nn.AdaptiveAvgPool2d((3, 5))
        self.image = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            image_pool,
        )
        if architecture == "spatial_v2":
            self.image_projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 3 * 5, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            image_dim = 128
        else:
            image_dim = 64
        self.lidar = nn.Sequential(nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.proprio = nn.Sequential(nn.Linear(57, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.language = nn.Sequential(nn.Embedding(257, 24, padding_idx=0), nn.Linear(24, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(image_dim + 32 + 64 + 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, horizon * 16),
        )
        self.stop_head = (
            nn.Sequential(nn.Linear(image_dim + 32 + 64 + 32, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1))
            if stop_head
            else None
        )

    def forward(
        self,
        rgb: torch.Tensor,
        lidar: torch.Tensor,
        proprio: torch.Tensor,
        language: torch.Tensor,
        return_stop: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image_feature = self.image(rgb)
        image_feature = self.image_projection(image_feature) if self.architecture == "spatial_v2" else image_feature.flatten(1)
        lidar_feature = self.lidar(lidar)
        proprio_feature = self.proprio(proprio)
        language_embedding = self.language[0](language)
        mask = language.ne(0).unsqueeze(-1)
        language_feature = (language_embedding * mask).sum(1) / mask.sum(1).clamp_min(1)
        language_feature = self.language[1:](language_feature)
        fused = torch.cat((image_feature, lidar_feature, proprio_feature, language_feature), dim=-1)
        action = self.head(fused).view(-1, self.horizon, 16)
        if return_stop:
            if self.stop_head is None:
                raise RuntimeError("return_stop=True requires stop_head=True")
            return action, self.stop_head(fused).squeeze(-1)
        return action
