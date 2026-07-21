"""Leg-only jumping skill for the M20 Pro."""

import torch

from .m20pro_locomotion_env import M20ProLocomotionEnv


class M20ProJumpEnv(M20ProLocomotionEnv):
    """Train vertical leg coordination with wheels position-locked."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.max_base_height = self.robot.data.root_pos_w[:, 2].clone()
        self._leg_ids, _ = self.robot.find_joints(".*_(hipx|hipy|knee)_joint")
        self._wheel_ids, _ = self.robot.find_joints(".*_wheel_joint")

    def _apply_action(self):
        # Actions are normalized leg joint targets; wheels remain position-locked.
        leg_targets = torch.clamp(self.actions * 0.8, min=-0.8, max=0.8)
        self.robot.set_joint_position_target(leg_targets, joint_ids=self._leg_ids)
        wheel_targets = self.robot.data.default_joint_pos[:, self._wheel_ids]
        self.robot.set_joint_position_target(wheel_targets, joint_ids=self._wheel_ids)

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        base_height = self.torso_position[:, 2]
        phase = self.episode_length_buf / self.max_episode_length
        reference_actions = torch.zeros_like(self.actions)
        reference_actions[:, 4:12] = torch.where(
            (phase < 0.30).unsqueeze(-1),
            torch.ones_like(reference_actions[:, 4:12]),
            -torch.ones_like(reference_actions[:, 4:12]),
        )
        previous_max = self.max_base_height.clone()
        self.max_base_height = torch.maximum(self.max_base_height, base_height)
        jump_progress = torch.clamp((self.max_base_height - self.cfg.initial_base_height) / 0.30, 0.0, 1.0)
        squat_reward = torch.exp(-torch.square(base_height - self.cfg.squat_base_height) / 0.01)
        height_tracking = torch.exp(-torch.square(self.max_base_height - self.cfg.target_jump_height) / 0.01)
        upward_velocity = torch.clamp(self.velocity[:, 2], min=0.0, max=2.0)
        upright = torch.clamp((self.up_proj - 0.75) / 0.25, min=0.0, max=1.0)
        leg_posture_cost = torch.sum(torch.square(self.dof_pos[:, :12]), dim=-1)
        leg_action_cost = torch.sum(torch.square(self.actions), dim=-1)
        reference_error = torch.mean(torch.square(self.actions - reference_actions), dim=-1)
        progress_reward = self.max_base_height - previous_max
        squat_phase = (phase < 0.30).to(torch.float32)
        takeoff_phase = ((phase >= 0.30) & (phase < 0.55)).to(torch.float32)
        flight_phase = (phase >= 0.55).to(torch.float32)
        reward = (
            1.0 * squat_phase * squat_reward
            + 4.0 * takeoff_phase * upward_velocity
            + 4.0 * flight_phase * jump_progress
            + 2.0 * flight_phase * height_tracking
            + 8.0 * takeoff_phase * progress_reward
            + 0.5 * upright
            + 2.0 * torch.exp(-reference_error / 0.25)
            - 0.02 * leg_posture_cost
            - 0.005 * leg_action_cost
        )
        return torch.where(self.reset_terminated, torch.full_like(reward, self.cfg.death_cost), reward)

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        phase = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)
        observations["policy"] = torch.cat((observations["policy"], phase), dim=-1)
        return observations

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.max_base_height[env_ids] = self.robot.data.root_pos_w[env_ids, 2]
