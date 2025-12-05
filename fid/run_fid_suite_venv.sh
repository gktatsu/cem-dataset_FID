#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_fid_suite_venv.sh REAL_DIR GEN_DIR [OPTIONS] [--] [EXTRA_ARGS...]

Runs compute_cem_fid.py and compute_normal_fid.py using a local Python
environment (venv) instead of Docker.

Constants:
  Weights dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/weights
  Results dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/results

Options (must be specified before "--"):
  --cem-backbone {cem500k|cem1.5m}  Select the CEM backbone (default: cem500k)
  --cem-weights PATH               Use a specific checkpoint inside the weights dir
  --venv PATH                      Path to Python venv (default: ../venv or $VIRTUAL_ENV)
  --python PYTHON                  Python executable to use (default: python)

EXTRA_ARGS after "--" are forwarded to both Python scripts. A "--data-volume"
argument equal to GEN_DIR is always appended automatically.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

REAL_DIR=$1
GEN_DIR=$2
shift 2

CEM_BACKBONE="cem500k"
CEM_WEIGHTS=""
VENV_PATH=""
PYTHON_CMD="python"
EXTRA_ARGS=()

# Check if we're already in a venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  VENV_PATH="$VIRTUAL_ENV"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cem-backbone)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --cem-backbone requires an argument" >&2
        exit 1
      fi
      CEM_BACKBONE=$2
      shift 2
      ;;
    --cem-weights)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --cem-weights requires an argument" >&2
        exit 1
      fi
      CEM_WEIGHTS=$2
      shift 2
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --venv requires an argument" >&2
        exit 1
      fi
      VENV_PATH=$2
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --python requires an argument" >&2
        exit 1
      fi
      PYTHON_CMD=$2
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$CEM_BACKBONE" in
  cem500k|cem1.5m)
    ;;
  *)
    echo "[ERROR] --cem-backbone must be cem500k or cem1.5m (got $CEM_BACKBONE)" >&2
    exit 1
    ;;
esac

for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --output-json|--num-workers)
      echo "[ERROR] --output-json と --num-workers はこのスクリプトでは指定できません。" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$REAL_DIR" ]]; then
  echo "[ERROR] REAL_DIR does not exist or is not a directory: $REAL_DIR" >&2
  exit 2
fi

if [[ ! -d "$GEN_DIR" ]]; then
  echo "[ERROR] GEN_DIR does not exist or is not a directory: $GEN_DIR" >&2
  exit 3
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/cem_fid"
mkdir -p "$RESULTS_DIR/normal_fid"

select_default_weights() {
  local backbone=$1
  case "$backbone" in
    cem500k)
      echo "cem500k_mocov2_resnet50_200ep.pth.tar"
      ;;
    cem1.5m)
      # Allow either the official filename or the balanced variant if present.
      if [[ -f "$WEIGHTS_DIR/cem1.5m_swav_resnet50_200ep_balanced.pth.tar" ]]; then
        echo "cem1.5m_swav_resnet50_200ep_balanced.pth.tar"
      else
        echo "cem15m_swav_resnet50_200ep.pth.tar"
      fi
      ;;
  esac
}

normalize_path() {
  local target=$1
  if command -v realpath >/dev/null 2>&1; then
    realpath "$target"
  else
    readlink -f "$target"
  fi
}

if [[ -z "$CEM_WEIGHTS" ]]; then
  default_name=$(select_default_weights "$CEM_BACKBONE")
  if [[ -n "$default_name" && -f "$WEIGHTS_DIR/$default_name" ]]; then
    CEM_WEIGHTS="$WEIGHTS_DIR/$default_name"
  fi
fi

CEM_WEIGHTS_REAL=""

if [[ -n "$CEM_WEIGHTS" ]]; then
  # If CEM_WEIGHTS is a relative path, treat it as relative to WEIGHTS_DIR
  if [[ "$CEM_WEIGHTS" != /* ]]; then
    CEM_WEIGHTS="$WEIGHTS_DIR/$CEM_WEIGHTS"
  fi
  
  CEM_WEIGHTS_REAL=$(normalize_path "$CEM_WEIGHTS")
  
  # Normalize WEIGHTS_DIR for comparison
  WEIGHTS_DIR_REAL=$(normalize_path "$WEIGHTS_DIR")
  
  if [[ ${CEM_WEIGHTS_REAL} != $WEIGHTS_DIR_REAL/* ]]; then
    echo "[ERROR] --cem-weights must reside within $WEIGHTS_DIR" >&2
    echo "[INFO] WEIGHTS_DIR: $WEIGHTS_DIR_REAL" >&2
    echo "[INFO] Provided weights path: $CEM_WEIGHTS_REAL" >&2
    exit 4
  fi
  if [[ ! -f "$CEM_WEIGHTS_REAL" ]]; then
    echo "[ERROR] Specified weights file not found: $CEM_WEIGHTS_REAL" >&2
    exit 4
  fi
fi

# Setup Python command
if [[ -n "$VENV_PATH" ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "[ERROR] venv directory not found: $VENV_PATH" >&2
    exit 5
  fi
  if [[ -f "$VENV_PATH/bin/python" ]]; then
    PYTHON_CMD="$VENV_PATH/bin/python"
  else
    echo "[ERROR] Python executable not found in venv: $VENV_PATH/bin/python" >&2
    exit 5
  fi
else
  # Try to find default venv
  if [[ -f "$SCRIPT_DIR/../venv/bin/python" ]]; then
    PYTHON_CMD="$SCRIPT_DIR/../venv/bin/python"
    echo "[INFO] Using venv at: $SCRIPT_DIR/../venv" >&2
  elif [[ -f "$SCRIPT_DIR/venv/bin/python" ]]; then
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
    echo "[INFO] Using venv at: $SCRIPT_DIR/venv" >&2
  fi
fi

# Verify Python is available
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_CMD" >&2
  echo "[INFO] Please specify a valid Python using --python or --venv" >&2
  exit 6
fi

echo "[INFO] Using Python: $PYTHON_CMD" >&2

# Generate timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M)
CEM_OUTPUT="$RESULTS_DIR/cem_fid/cem_fid_${TIMESTAMP}.json"
NORMAL_OUTPUT="$RESULTS_DIR/normal_fid/normal_fid_${TIMESTAMP}.json"

# Prepare common arguments
COMMON_ARGS=("${EXTRA_ARGS[@]}" "--num-workers" "0" "--data-volume" "$GEN_DIR")

# Prepare CEM FID arguments
CEM_ARGS=("${COMMON_ARGS[@]}" "--output-json" "$CEM_OUTPUT" "--backbone" "$CEM_BACKBONE")
if [[ -n "$CEM_WEIGHTS_REAL" ]]; then
  CEM_ARGS+=("--weights-path" "$CEM_WEIGHTS_REAL")
fi

# Prepare Normal FID arguments
NORMAL_ARGS=("${COMMON_ARGS[@]}" "--output-json" "$NORMAL_OUTPUT")

# Change to repository root
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Running CEM FID computation..." >&2
"$PYTHON_CMD" "fid/compute_cem_fid.py" "$REAL_DIR" "$GEN_DIR" "${CEM_ARGS[@]}"

echo "[INFO] Running Normal FID computation..." >&2
"$PYTHON_CMD" "fid/compute_normal_fid.py" "$REAL_DIR" "$GEN_DIR" "${NORMAL_ARGS[@]}"

echo "[INFO] Done!" >&2
echo "[INFO] CEM FID results: $CEM_OUTPUT" >&2
echo "[INFO] Normal FID results: $NORMAL_OUTPUT" >&2
