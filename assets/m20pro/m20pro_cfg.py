"""Isaac Lab configuration for the DEEP Robotics M20 Pro."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = Path(__file__).with_name("m20pro.usd")

M20PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_(hipx|hipy|knee)_joint"],
            effort_limit=76.4,
            saturation_effort=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            saturation_effort=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)

M20PRO_JUMP_CFG = M20PRO_CFG.replace(
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_(hipx|hipy|knee)_joint"],
            stiffness=80.0,
            damping=8.0,
            effort_limit_sim=76.4,
            velocity_limit_sim=22.4,
        ),
        "wheels_locked": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=120.0,
            damping=12.0,
            effort_limit_sim=21.6,
            velocity_limit_sim=79.3,
        ),
    }
)
