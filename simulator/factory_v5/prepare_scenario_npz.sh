#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <scenario-dir> <fds-result-dir> [fds-input]" >&2
    exit 2
fi

SCENARIO_DIR="$(realpath -m "$1")"
FDS_RESULT_DIR="$(realpath "$2")"
FDS_INPUT="$(realpath "${3:-${SCRIPT_DIR}/factory_v5.fds}")"
OUTPUT_DIR="${SCENARIO_DIR}/processed"
mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/pack_temp3d_to_npz.py" \
  --from-fds --fds-result-dir "${FDS_RESULT_DIR}" \
  --fds-file "${FDS_INPUT}" \
  --output "${OUTPUT_DIR}/fds_temperature_3d_timeseries.npz"
python3 "${SCRIPT_DIR}/pack_co_to_npz.py" \
  --fds-result-dir "${FDS_RESULT_DIR}" \
  --output "${OUTPUT_DIR}/fds_co_2d_timeseries.npz"

echo "Scenario runtime archives created in ${OUTPUT_DIR}"
