# M20Pro Sim Scripts

These scripts are local simulation helpers. They do not SSH to 104, install systemd units, record real bags, or control the robot.

```bash
./scripts/start_sim.sh true
./scripts/status_sim.sh
./scripts/stop_sim.sh
```

Use `false` as the first argument to start without RViz:

```bash
./scripts/start_sim.sh false
```

## VLA / Isaac Lab environment

Training outputs are stored on the mounted 2 TB disk. Repair the workspace
links after a reboot if needed:

```bash
./scripts/prepare_output_storage.sh
```

The current disk is `/dev/sdb2` (UUID
`b9cbb43d-5119-4328-99d9-10f7c0d91e37`). Mount it with:

```bash
udisksctl mount -b /dev/disk/by-uuid/b9cbb43d-5119-4328-99d9-10f7c0d91e37
```

The embodied-learning environment is isolated in the `m20pro-vla` Conda
environment. It uses Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2, PyTorch
2.7/CUDA 12.8, and RSL-RL.

```bash
./scripts/install_vla_env.sh
source ./scripts/activate_vla_env.sh
./scripts/check_vla_env.sh
```

Isaac Sim needs most of the RTX 3060's 12 GB VRAM. Stop other GPU training
jobs before launching it. The first simulator launch requires accepting the
NVIDIA Omniverse license in the terminal.

Convert and validate the M20 Pro asset:

```bash
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
```

The first locomotion task configuration can be checked with:

```bash
TERM=xterm python scripts/check_m20pro_task.py --headless
```

Run the first real environment reset/step smoke test:

```bash
./scripts/smoke_m20pro_locomotion.sh --num-envs 1 --steps 4
```

Record a trained policy without opening the Isaac Sim GUI:

```bash
./scripts/play_m20pro_ppo.sh \
  --checkpoint logs/rsl_rl/m20pro_locomotion_v2/model_299.pt \
  --num-envs 1 --steps 400 --video \
  --video-dir videos/m20pro_locomotion_v2
```

Smoke-test the leg-only jump skill:

```bash
./scripts/smoke_m20pro_jump.sh --num-envs 4 --steps 24
```
