#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${M20PRO_VLA_ENV:-m20pro-vla}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${WORKSPACE}/.deps/IsaacLab"
WHEEL_DIR="${WORKSPACE}/.deps/wheels"
CONDA_EXE="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
ISAACLAB_TAG="v2.3.2"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda was not found at ${CONDA_EXE}. Set CONDA_EXE explicitly." >&2
  exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_EXE}" create -y -n "${ENV_NAME}" python=3.11
fi

if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${ISAACLAB_DIR}")"
  git clone --branch "${ISAACLAB_TAG}" --depth 1 \
    https://gitee.com/mirrors/IsaacLab.git "${ISAACLAB_DIR}"
fi

ENV_PYTHON="${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  ENV_PYTHON="$("${CONDA_EXE}" run -n "${ENV_NAME}" command -v python)"
fi

"${ENV_PYTHON}" -m pip install --upgrade pip \
  "setuptools==80.9.0" "wheel==0.45.1"
"${ENV_PYTHON}" -m pip install --upgrade \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

# NVIDIA's extension-cache wheels are several GB each. Download them with
# resume support before asking pip to resolve the Isaac Sim metapackage.
mkdir -p "${WHEEL_DIR}"
for package in \
  isaacsim_extscache_kit \
  isaacsim_extscache_kit_sdk \
  isaacsim_extscache_physics; do
  wheel="${package}-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
  wget --continue --progress=bar:force:noscroll \
    "https://pypi.nvidia.cn/${package//_/-}/${wheel}" \
    -O "${WHEEL_DIR}/${wheel}"
done
"${ENV_PYTHON}" -m pip install "${WHEEL_DIR}"/*.whl

"${ENV_PYTHON}" -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
# Install only the extensions used by this project. The upstream all-in-one
# installer also installs Mimic/Jupyter, whose psutil requirement conflicts
# with Isaac Sim 5.1's strict psutil pin.
"${ENV_PYTHON}" -m pip uninstall -y isaaclab_mimic ipython ipywidgets >/dev/null 2>&1 || true
"${ENV_PYTHON}" -m pip install --no-build-isolation "flatdict==4.0.1"
"${ENV_PYTHON}" -m pip install -e "${ISAACLAB_DIR}/source/isaaclab"
"${ENV_PYTHON}" -m pip install -e "${ISAACLAB_DIR}/source/isaaclab_assets"
"${ENV_PYTHON}" -m pip install -e "${ISAACLAB_DIR}/source/isaaclab_tasks"
"${ENV_PYTHON}" -m pip install -e "${ISAACLAB_DIR}/source/isaaclab_rl[rsl_rl]"
"${ENV_PYTHON}" -m pip install \
  "onnx==1.18.0" "onnxruntime==1.27.0" "packaging==23.0" "psutil==5.9.8" \
  "starlette==0.45.3" "typing_extensions==4.12.2"

# Isaac Lab 2.3.2 declares starlette 0.49.1 while Isaac Sim 5.1's pinned
# FastAPI requires starlette <0.46. Runtime compatibility takes precedence.
check_output="$("${ENV_PYTHON}" -m pip check 2>&1 || true)"
unexpected="$(printf '%s\n' "${check_output}" | grep -vE \
  '^isaaclab 0\.54\.2 has requirement starlette==0\.49\.1, but you have starlette 0\.45\.3\.$|^$' || true)"
if [[ -n "${unexpected}" ]]; then
  printf '%s\n' "${unexpected}" >&2
  exit 1
fi

echo "Environment ${ENV_NAME} is installed."
echo "Activate it with: source ${WORKSPACE}/scripts/activate_vla_env.sh"
