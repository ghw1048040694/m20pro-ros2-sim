"""Search a small library of symmetric M20 jump motion primitives.

This is an expert-data bootstrap, not PPO and not a reward-driven policy
trainer.  Each candidate is a short squat -> thrust -> settle joint-target
sequence.  Candidates are evaluated in parallel and ranked by physical height,
survival and displacement so the best one can seed imitation learning.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--top-k", type=int, default=8)
parser.add_argument(
    "--json-output",
    type=Path,
    default=Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_jump_primitive_search.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 30:
    parser.error("--steps must be at least 30")
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from tasks.m20pro_locomotion import M20ProJumpEnv, M20ProJumpEnvCfg  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    front_hipy_sign: float
    rear_hipy_sign: float
    knee_sign: float
    amplitude: float
    squat_steps: int
    thrust_steps: int
    thrust_multiplier: float


def candidates() -> list[Candidate]:
    values: list[Candidate] = []
    for front, rear in ((1.0, 1.0), (-1.0, -1.0)):
        for knee in (-1.0, 1.0):
            for amplitude in (0.6, 0.8, 1.0):
                for phase_steps in (6, 10, 15):
                    for thrust_multiplier in (0.0, -0.5, -1.0):
                        values.append(
                            Candidate(
                                front,
                                rear,
                                knee,
                                amplitude,
                                phase_steps,
                                phase_steps,
                                thrust_multiplier,
                            )
                        )
    return values


def action_batch(specs: list[Candidate], step: int, device: torch.device) -> torch.Tensor:
    actions = torch.zeros((len(specs), 12), device=device)
    for env_id, spec in enumerate(specs):
        if step < spec.squat_steps:
            phase_scale = 1.0
        elif step < spec.squat_steps + spec.thrust_steps:
            phase_scale = spec.thrust_multiplier
        else:
            phase_scale = 0.0
        actions[env_id, 4:6] = phase_scale * spec.front_hipy_sign * spec.amplitude
        actions[env_id, 6:8] = phase_scale * spec.rear_hipy_sign * spec.amplitude
        actions[env_id, 8:12] = phase_scale * spec.knee_sign * spec.amplitude
    return actions


def main() -> int:
    specs = candidates()
    cfg = M20ProJumpEnvCfg()
    cfg.scene.num_envs = len(specs)
    env = M20ProJumpEnv(cfg)
    try:
        env.reset()
        initial_pos = env.robot.data.root_pos_w[:, :3].clone()
        max_height = initial_pos[:, 2].clone()
        min_height = initial_pos[:, 2].clone()
        done_count = torch.zeros(len(specs), dtype=torch.int32, device=env.device)
        for step in range(args.steps):
            actions = action_batch(specs, step, env.device)
            _, _, terminated, truncated, _ = env.step(actions)
            root_pos = env.robot.data.root_pos_w[:, :3]
            max_height = torch.maximum(max_height, root_pos[:, 2])
            min_height = torch.minimum(min_height, root_pos[:, 2])
            done_count += (terminated | truncated).to(torch.int32)
        final_pos = env.robot.data.root_pos_w[:, :3]
        rows = []
        for i, spec in enumerate(specs):
            row = asdict(spec)
            row.update(
                {
                    "candidate_id": i,
                    "max_root_height": float(max_height[i].item()),
                    "min_root_height": float(min_height[i].item()),
                    "final_displacement": float(torch.linalg.vector_norm(final_pos[i] - initial_pos[i]).item()),
                    "final_x_displacement": float((final_pos[i, 0] - initial_pos[i, 0]).item()),
                    "done_count": int(done_count[i].item()),
                    "survived": bool(done_count[i].item() == 0 and min_height[i].item() >= 0.35),
                }
            )
            rows.append(row)
        rows.sort(key=lambda row: (row["survived"], row["max_root_height"], -row["final_displacement"]), reverse=True)
        payload = {
            "format": "m20pro_jump_primitive_search_v0",
            "steps": args.steps,
            "candidate_count": len(rows),
            "source": "physics rollout; no PPO/reward optimization",
            "results": rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[M20PRO-JUMP-SEARCH] candidates={len(rows)} steps={args.steps}", flush=True)
        for row in rows[: args.top_k]:
            print(
                "[M20PRO-JUMP-SEARCH] "
                f"id={row['candidate_id']} survived={row['survived']} "
                f"max_z={row['max_root_height']:.4f} min_z={row['min_root_height']:.4f} "
                f"dx={row['final_x_displacement']:.4f} "
                f"front={row['front_hipy_sign']:+.0f} rear={row['rear_hipy_sign']:+.0f} "
                f"knee={row['knee_sign']:+.0f} amp={row['amplitude']:.1f} "
                f"phase={row['squat_steps']}/{row['thrust_steps']} thrust={row['thrust_multiplier']:+.1f}",
                flush=True,
            )
        return 0
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
