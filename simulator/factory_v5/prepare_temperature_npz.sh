#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_DIR="${1:-${SCRIPT_DIR}/scenarios/baseline}"
FDS_RESULT_DIR="${FDS_RESULT_DIR:-${SCENARIO_DIR}/fds_result}"
mkdir -p "${SCENARIO_DIR}/processed"
python3 "${SCRIPT_DIR}/pack_temp3d_to_npz.py" \
  --from-fds \
  --fds-result-dir "${FDS_RESULT_DIR}" \
  --fds-file "${SCRIPT_DIR}/factory_v5.fds" \
  --output "${SCENARIO_DIR}/processed/fds_temperature_3d_timeseries.npz"
