"""M20 Pro locomotion task definitions for Isaac Lab."""

from .m20pro_locomotion_env import M20ProLocomotionEnv
from .m20pro_locomotion_env_cfg import M20ProLocomotionEnvCfg
from .jump_env import M20ProJumpEnv
from .jump_env_cfg import M20ProJumpEnvCfg

import gymnasium as gym

from . import agents

gym.register(
    id="M20Pro-Locomotion-Flat-v0",
    entry_point=f"{__name__}.m20pro_locomotion_env:M20ProLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m20pro_locomotion_env_cfg:M20ProLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20ProLocomotionPPORunnerCfg",
    },
)

gym.register(
    id="M20Pro-Jump-Direct-v0",
    entry_point=f"{__name__}.jump_env:M20ProJumpEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.jump_env_cfg:M20ProJumpEnvCfg"},
)

__all__ = ["M20ProLocomotionEnv", "M20ProLocomotionEnvCfg"]
