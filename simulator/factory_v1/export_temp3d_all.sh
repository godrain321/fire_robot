#!/usr/bin/env bash
set -e

cd ~/Firesim/factory_v1

source ~/FDS/FDS6/bin/FDS6VARS.sh
export I_MPI_FABRICS=shm

CHID="factory_v1"
VAR_INDEX=1

OUT_DIR="csv_temp3d"
mkdir -p "$OUT_DIR"

# T_END=90이므로 0~89초 구간을 [t, t+1] 평균으로 추출
for T_START in $(seq 0 89)
do
    T_END=$((T_START + 1))
    T_PAD=$(printf "%03d" "$T_START")

    OUT_FILE="${CHID}_temp_3d_t${T_PAD}.csv"

    echo "Exporting ${OUT_FILE} from ${T_START}s to ${T_END}s..."

    rm -f "$OUT_FILE"
    rm -f "${OUT_DIR}/${OUT_FILE}"

    fds2ascii << EOF
$CHID
2
1
n
$T_START $T_END
1
$VAR_INDEX
$OUT_FILE
EOF

    mv "$OUT_FILE" "$OUT_DIR/$OUT_FILE"
done

echo "Done. CSV files are saved in ${OUT_DIR}/"
