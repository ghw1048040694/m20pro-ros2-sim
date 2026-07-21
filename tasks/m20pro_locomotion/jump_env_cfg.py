"""Configuration for the M20 Pro leg-only jumping skill."""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from .m20pro_locomotion_env_cfg import M20ProLocomotionEnvCfg


@configclass
class M20ProJumpEnvCfg(M20ProLocomotionEnvCfg):
    episode_length_s = 4.0
    action_space = 12
    observation_space = 56
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
    termination_height = 0.35
    target_forward_velocity = 0.0
    initial_base_height = 0.62
    target_jump_height = 0.80
