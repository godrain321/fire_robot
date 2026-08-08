#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHID="factory_v3_cat"
# Current factory_v3 FDS output contains t=0..200 s (201 frames).
T_END=201
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
for ((t_start=0; t_start<T_END; t_start++)); do
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
