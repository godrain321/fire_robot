#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHID="factory_v5_cat"
FDS_INPUT="${SCRIPT_DIR}/factory_v5.fds"
if [[ ! -f "${FDS_INPUT}" ]]; then
    echo "FDS input not found: ${FDS_INPUT}" >&2
    exit 1
fi
# Include both endpoints: T_END=600 produces frames t=0..600.
LAST_TIME="$({ sed -n 's/.*T_END[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${FDS_INPUT}" || true; } | head -n 1)"
if [[ -z "${LAST_TIME}" ]]; then
    echo "Integer T_END not found in ${FDS_INPUT}" >&2
    exit 1
fi
START_TIME="${START_TIME:-0}"
VAR_INDEX=1
OUT_DIR="${SCRIPT_DIR}/csv_temp3d"
FDS2ASCII="${FDS2ASCII:-/home/park/FDS/FDS6/bin/fds2ascii}"

if [[ ! -x "${FDS2ASCII}" ]]; then
    echo "fds2ascii executable not found: ${FDS2ASCII}" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/${CHID}.smv" ]]; then
    echo "FDS result metadata not found: ${SCRIPT_DIR}/${CHID}.smv" >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"
cd "${SCRIPT_DIR}"
export I_MPI_FABRICS=shm

# Same verified fds2ascii menu sequence as factory_v1. Each output is the
# [t,t+1] interval for TEMP_3D (slice variable 1), named by numeric start time.
for ((t_start=START_TIME; t_start<=LAST_TIME; t_start++)); do
    t_stop=$((t_start + 1))
    printf -v t_padded '%03d' "${t_start}"
    output_name="${CHID}_temp_3d_t${t_padded}.csv"
    output_path="${OUT_DIR}/${output_name}"
    if [[ -e "${output_path}" ]]; then
        echo "Refusing to overwrite existing CSV: ${output_path}" >&2
        exit 1
    fi
    echo "Exporting ${output_name}: ${t_start}s <= t <= ${t_stop}s"
    "${FDS2ASCII}" <<EOF
${CHID}
2
1
n
${t_start} ${t_stop}
1
${VAR_INDEX}
${output_name}
EOF
    if [[ ! -s "${SCRIPT_DIR}/${output_name}" ]]; then
        echo "fds2ascii did not create a non-empty ${output_name}" >&2
        exit 1
    fi
    mv "${SCRIPT_DIR}/${output_name}" "${output_path}"
done

echo "Temperature CSV export complete: ${OUT_DIR}"
