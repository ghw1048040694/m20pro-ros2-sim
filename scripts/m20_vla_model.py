"""Shared compact M20 multimodal action-chunk model."""

from __future__ import annotations

import torch
from torch import nn


class M20VLAActionChunk(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.lidar = nn.Sequential(nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.proprio = nn.Sequential(nn.Linear(57, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.language = nn.Sequential(nn.Embedding(257, 24, padding_idx=0), nn.Linear(24, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(64 + 32 + 64 + 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, horizon * 16),
        )

    def forward(self, rgb: torch.Tensor, lidar: torch.Tensor, proprio: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        image_feature = self.image(rgb).flatten(1)
        lidar_feature = self.lidar(lidar)
        proprio_feature = self.proprio(proprio)
        language_embedding = self.language[0](language)
        mask = language.ne(0).unsqueeze(-1)
        language_feature = (language_embedding * mask).sum(1) / mask.sum(1).clamp_min(1)
        language_feature = self.language[1:](language_feature)
        fused = torch.cat((image_feature, lidar_feature, proprio_feature, language_feature), dim=-1)
        return self.head(fused).view(-1, self.horizon, 16)
