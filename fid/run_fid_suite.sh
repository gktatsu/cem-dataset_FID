#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_fid_suite.sh REAL_DIR GEN_DIR [--] [EXTRA_ARGS...]

Runs CEM-based FID followed by Inception-based FID on the same directories.
  REAL_DIR  Directory containing real/reference images
  GEN_DIR   Directory containing generated images to evaluate

Optional EXTRA_ARGS after "--" are forwarded to both Python scripts.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

REAL_DIR=$1
GEN_DIR=$2
shift 2
EXTRA_ARGS=("$@")

if [[ ! -d "$REAL_DIR" ]]; then
  echo "[ERROR] REAL_DIR does not exist or is not a directory: $REAL_DIR" >&2
  exit 2
fi

if [[ ! -d "$GEN_DIR" ]]; then
  echo "[ERROR] GEN_DIR does not exist or is not a directory: $GEN_DIR" >&2
  exit 3
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CEM_SCRIPT="$SCRIPT_DIR/compute_cem_fid.py"
NORMAL_SCRIPT="$SCRIPT_DIR/compute_normal_fid.py"

if [[ ! -f "$CEM_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $CEM_SCRIPT" >&2
  exit 4
fi

if [[ ! -f "$NORMAL_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $NORMAL_SCRIPT" >&2
  exit 5
fi

python "$CEM_SCRIPT" "$REAL_DIR" "$GEN_DIR" "${EXTRA_ARGS[@]}"
python "$NORMAL_SCRIPT" "$REAL_DIR" "$GEN_DIR" "${EXTRA_ARGS[@]}"
