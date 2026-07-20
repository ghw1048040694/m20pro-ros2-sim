#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh" >/dev/null

RESULTS_DIR="${M20PRO_SIM_WS}/logs/m20pro_env_benchmark"
mkdir -p "${RESULTS_DIR}"
printf 'envs\tresult\tmax_gpu_memory\n' | tee "${RESULTS_DIR}/summary.tsv"

for envs in 16 32 64 128 256; do
  log="${RESULTS_DIR}/envs_${envs}.log"
  echo "[BENCH] testing ${envs} environments"
  TERM=xterm timeout --signal=INT --kill-after=10s 90s \
    python "${SCRIPT_DIR}/smoke_m20pro_locomotion.py" \
      --headless --num-envs "${envs}" --steps 4 >"${log}" 2>&1
  rc=$?
  if rg -q '\[M20PRO-RL\] reset/step passed' "${log}"; then
    result=pass
  else
    result=fail
  fi
  gpu_mem=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)
  printf '%s\t%s\t%s\n' "${envs}" "${result}(rc=${rc})" "${gpu_mem:-none}" | tee -a "${RESULTS_DIR}/summary.tsv"
done

echo "[BENCH] results: ${RESULTS_DIR}/summary.tsv"
