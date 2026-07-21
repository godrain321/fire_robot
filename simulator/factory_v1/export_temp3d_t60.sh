#!/usr/bin/env bash

cd ~/Firesim/factory_v1

source ~/FDS/FDS6/bin/FDSVARS.sh
export I_MPI_FABRICS=shm

CHID="factory_v1"
T_START=60
T_END=61
VAR_INDEX=1
OUT_FILE="factory_v1_temp_3d_t60.csv"

# 기존 파일이 있으면 fds2ascii가 덮어쓸지 물어볼 수 있으니 미리 삭제
rm -f "$OUT_FILE"

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
