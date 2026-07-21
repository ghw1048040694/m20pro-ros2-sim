"""Direct-RL locomotion environment specialized for the M20 Pro."""

import torch

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class M20ProLocomotionEnv(LocomotionEnv):
    """Torque-controlled baseline used before adding vision and language."""

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        # Continuous wheel joints have unbounded USD limits. The generic
        # locomotion normalizer would evaluate Inf / Inf for their positions.
        self.dof_pos_scaled[:, 12:] = 0.0

    def _get_rewards(self) -> torch.Tensor:
        """Reward commanded forward motion while retaining an upright chassis."""
        self._compute_intermediate_values()
        velocity_error = torch.square(self.vel_loc[:, 0] - self.cfg.target_forward_velocity)
        forward_tracking = torch.exp(-velocity_error / 0.25)
        upright = torch.clamp((self.up_proj - 0.8) / 0.2, min=0.0, max=1.0)
        lateral_velocity = torch.square(self.vel_loc[:, 1])
        vertical_velocity = torch.square(self.vel_loc[:, 2])
        angular_velocity = torch.sum(torch.square(self.angvel_loc), dim=-1)
        action_cost = torch.sum(torch.square(self.actions), dim=-1)
        reward = (
            2.0 * forward_tracking
            + 0.5 * upright
            + 0.05
            - 0.10 * lateral_velocity
            - 0.20 * vertical_velocity
            - 0.02 * angular_velocity
            - 0.005 * action_cost
        )
        return torch.where(self.reset_terminated, torch.full_like(reward, self.cfg.death_cost), reward)
