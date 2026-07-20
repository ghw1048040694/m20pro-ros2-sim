"""Direct-RL locomotion environment reused with the M20 Pro asset."""

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class M20ProLocomotionEnv(LocomotionEnv):
    """Torque-controlled baseline used before adding vision and language."""

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        # Continuous wheel joints have unbounded USD limits. The generic
        # locomotion normalizer would evaluate Inf / Inf for their positions.
        self.dof_pos_scaled[:, 12:] = 0.0
