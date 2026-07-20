#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"

python - <<'PY'
from importlib.metadata import version

import torch

print(f"python: {__import__('sys').version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"compute_capability: {torch.cuda.get_device_capability(0)}")
for package in ("isaacsim", "isaaclab", "isaaclab_tasks", "rsl-rl-lib"):
    try:
        print(f"{package}: {version(package)}")
    except Exception as exc:
        print(f"{package}: unavailable ({exc})")
PY

nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
