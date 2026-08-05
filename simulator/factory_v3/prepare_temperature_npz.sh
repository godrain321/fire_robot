#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/export_temp3d_all.sh"
python3 "${SCRIPT_DIR}/pack_temp3d_to_npz.py"

