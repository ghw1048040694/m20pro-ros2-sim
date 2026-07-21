"""Configuration for the M20 Pro leg-only jumping skill."""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from .m20pro_locomotion_env_cfg import M20ProLocomotionEnvCfg
from assets.m20pro import M20PRO_JUMP_CFG


@configclass
class M20ProJumpEnvCfg(M20ProLocomotionEnvCfg):
    episode_length_s = 2.0
    action_scale = 1.0
    action_space = 12
    observation_space = 57
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
    robot = M20PRO_JUMP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    termination_height = 0.35
    target_forward_velocity = 0.0
    initial_base_height = 0.62
    target_jump_height = 0.80
    squat_base_height = 0.50
