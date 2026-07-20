"""Flat-ground M20 Pro task configuration.

This is deliberately a small proprioceptive baseline. Camera/LiDAR tensors and
language embeddings will be added as additional observation groups after the
locomotion policy is stable.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from assets.m20pro import M20PRO_CFG


@configclass
class M20ProLocomotionEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.35
    action_space = 16
    observation_space = 60
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
    )
    # RTX 3060-friendly default after the 16-256 environment smoke benchmark.
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
    robot = M20PRO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # LocomotionEnv's simple target is +X. Keep torque ranges close to the URDF limits.
    joint_gears = [76.4] * 12 + [21.6] * 4
    heading_weight = 0.5
    up_weight = 0.2
    actions_cost_scale = 0.01
    energy_cost_scale = 0.005
    dof_vel_scale = 0.1
    death_cost = -2.0
    alive_reward_scale = 0.2
    angular_velocity_scale = 0.25
    termination_height = 0.35
