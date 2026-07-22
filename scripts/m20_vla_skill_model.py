"""Compact high-level VLA model for the M20 two-layer controller."""

from __future__ import annotations

import torch
from torch import nn


SKILL_NAMES = ("forward", "backward", "left", "right", "stop", "search", "jump")
COMMAND_SCALE = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)


class M20VLASkillPolicy(nn.Module):
    """Predict a navigation skill and the command consumed by the M20 expert.

    The command is normalized to [-1, 1] internally and maps to
    ``[forward, lateral, yaw]`` in the native M20 policy protocol.  ``search``
    and ``jump`` are part of the stable interface but are only trainable after
    verified demonstrations are added.
    """

    def __init__(
        self,
        architecture: str = "spatial_v2",
        search_head: bool = False,
        target_head: bool = False,
    ):
        super().__init__()
        if architecture not in {"global_v1", "spatial_v2"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        self.search_head_enabled = search_head
        self.target_head_enabled = target_head
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
        self.lidar = nn.Sequential(
            nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.proprio = nn.Sequential(
            nn.Linear(57, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64), nn.GELU()
        )
        self.language = nn.Sequential(nn.Embedding(257, 24, padding_idx=0), nn.Linear(24, 32), nn.GELU())
        fused_dim = image_dim + 32 + 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.command_head = nn.Linear(256, 3)
        self.skill_head = nn.Linear(256, len(SKILL_NAMES))
        self.search_head = (
            nn.Sequential(nn.Linear(24, 32), nn.GELU(), nn.Linear(32, 1)) if search_head else None
        )
        self.target_head = (
            nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 2))
            if target_head
            else None
        )

    def forward(
        self,
        rgb: torch.Tensor,
        lidar: torch.Tensor,
        proprio: torch.Tensor,
        language: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_feature = self.image(rgb)
        if self.architecture == "spatial_v2":
            image_feature = self.image_projection(image_feature)
        else:
            image_feature = image_feature.flatten(1)
        lidar_feature = self.lidar(lidar)
        proprio_feature = self.proprio(proprio)
        language_embedding = self.language[0](language)
        mask = language.ne(0).unsqueeze(-1)
        language_feature = (language_embedding * mask).sum(1) / mask.sum(1).clamp_min(1)
        language_feature = self.language[1:](language_feature)
        fused = self.fusion(torch.cat((image_feature, lidar_feature, proprio_feature, language_feature), dim=-1))
        command = torch.tanh(self.command_head(fused))
        skill_logits = self.skill_head(fused)
        if self.search_head is None and self.target_head is None:
            return command, skill_logits
        outputs = [command, skill_logits]
        if self.search_head is not None:
            search_feature = (language_embedding * mask).sum(1) / mask.sum(1).clamp_min(1)
            outputs.append(self.search_head(search_feature).squeeze(-1))
        if self.target_head is not None:
            outputs.append(self.target_head(fused).squeeze(-1))
        return tuple(outputs)
