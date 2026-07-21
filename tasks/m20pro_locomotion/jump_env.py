"""Leg-only jumping skill for the M20 Pro."""

import torch

from .m20pro_locomotion_env import M20ProLocomotionEnv


class M20ProJumpEnv(M20ProLocomotionEnv):
    """Train vertical leg power with wheel torques disabled."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.max_base_height = self.robot.data.root_pos_w[:, 2].clone()

    def _apply_action(self):
        # The first 12 actions are leg torques. Wheels remain passive/locked.
        forces = self.action_scale * self.joint_gears[:12] * self.actions
        wheel_forces = torch.zeros((self.num_envs, 4), device=self.sim.device)
        self.robot.set_joint_effort_target(torch.cat((forces, wheel_forces), dim=-1), joint_ids=self._joint_dof_idx)

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        base_height = self.torso_position[:, 2]
        previous_max = self.max_base_height.clone()
        self.max_base_height = torch.maximum(self.max_base_height, base_height)
        jump_progress = torch.clamp((self.max_base_height - self.cfg.initial_base_height) / 0.30, 0.0, 1.0)
        height_tracking = torch.exp(-torch.square(self.max_base_height - self.cfg.target_jump_height) / 0.01)
        upward_velocity = torch.clamp(self.velocity[:, 2], min=0.0, max=2.0)
        upright = torch.clamp((self.up_proj - 0.75) / 0.25, min=0.0, max=1.0)
        leg_posture_cost = torch.sum(torch.square(self.dof_pos[:, :12]), dim=-1)
        leg_action_cost = torch.sum(torch.square(self.actions), dim=-1)
        progress_reward = self.max_base_height - previous_max
        reward = (
            4.0 * jump_progress
            + 1.0 * height_tracking
            + 1.0 * upward_velocity
            + 0.5 * upright
            + 8.0 * progress_reward
            - 0.02 * leg_posture_cost
            - 0.005 * leg_action_cost
        )
        return torch.where(self.reset_terminated, torch.full_like(reward, self.cfg.death_cost), reward)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.max_base_height[env_ids] = self.robot.data.root_pos_w[env_ids, 2]
