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

Record NVIDIA's published Isaac Lab Go2 expert with a video and a 235-dim
observation trajectory. The checkpoint cache, HDF5 data, and MP4 are placed on
the 2 TB output disk:

```bash
./scripts/record_public_go2_expert.sh \
  --steps 400 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_go2_rough_v0
```

Retarget its 12 joint targets to the M20 leg order. The output is marked
`validated=False` until a M20 video replay confirms signs and offsets:

```bash
python scripts/retarget_go2_to_m20.py INPUT.h5 OUTPUT.h5
```

Replay the retargeted actions on the M20. This is a third-person video
calibration pass and reports root-height/displacement diagnostics:

```bash
./scripts/play_m20_retargeted.sh \
  --actions-h5 OUTPUT.h5 --steps 200 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_retargeted_v0
```

Validate the released Robot Parkour Learning Go1 visual checkpoint without
launching Isaac Sim:

```bash
python scripts/validate_public_parkour_checkpoint.py \
  --json-output /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/public_parkour_checkpoint_validation.json
```

Replay the native Go1 public parkour policy with a mandatory third-person MP4.
The public checkpoint was trained for roughly 0.4--0.45 m parkour obstacles;
this command is an IsaacGym-to-IsaacLab protocol diagnostic, not an M20 policy:

```bash
./scripts/play_public_go1_parkour.sh \
  --policy-mode skill --steps 200 --command-x -1.0 \
  --obstacle-height 0.45 --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_parkour_go1
```

Run the non-visual Go1 walk checkpoint as a dynamics/action-protocol control:

```bash
./scripts/play_public_go1_parkour.sh \
  --policy-mode walk --steps 200 --command-x -1.0 --obstacle-x 5.0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_parkour_go1_walk
```
