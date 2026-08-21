#!/usr/bin/env bash
set -o pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY_DIR="$(cd "${SCENARIO_DIR}/../.." && pwd)"
cd "${SCENARIO_DIR}"

find . -maxdepth 1 -type f \( \
  -name 'FDS_FAILED' -o -name 'NPZ_FAILED' -o -name 'NPZ_COMPLETE' \
\) -delete

source /home/park/FDS/FDS6/bin/FDS6VARS.sh
export I_MPI_FABRICS=shm
export OMP_STACKSIZE=1G
export OMP_NUM_THREADS=4

echo "=== SCENARIO3 FDS started: $(date --iso-8601=seconds) ==="
fds_openmp scenario3.fds 2>&1 | tee fds_run.log
fds_status=${PIPESTATUS[0]}
if [[ ${fds_status} -ne 0 ]]; then
  echo "=== FDS FAILED: exit ${fds_status}; NPZ conversion skipped ==="
  touch FDS_FAILED
  exit "${fds_status}"
fi

echo "=== FDS complete; organizing raw results ==="
find . -maxdepth 1 -type f -name 'scenario3_cat*' \
  -exec mv -t fds_result -- {} +

cd "${FACTORY_DIR}"
./prepare_scenario_npz.sh \
  scenarios/scenario3 \
  scenarios/scenario3/fds_result \
  scenarios/scenario3/scenario3.fds \
  2>&1 | tee scenarios/scenario3/npz_conversion.log
npz_status=${PIPESTATUS[0]}
if [[ ${npz_status} -ne 0 ]]; then
  touch scenarios/scenario3/NPZ_FAILED
  echo "=== NPZ FAILED: exit ${npz_status} ==="
  exit "${npz_status}"
fi

touch scenarios/scenario3/NPZ_COMPLETE
echo "=== TEMPERATURE AND CO NPZ COMPLETE: $(date --iso-8601=seconds) ==="
