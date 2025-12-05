#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_fid_suite_docker.sh REAL_DIR GEN_DIR [OPTIONS] [--] [EXTRA_ARGS...]

Runs compute_cem_fid.py and compute_normal_fid.py inside a fixed Docker
container environment using the host directories specified below.

Constants:
  Docker image : cem-fid
  Weights dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/weights
  Results dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/results

Options (must be specified before "--"):
  --cem-backbone {cem500k|cem1.5m}  Select the CEM backbone (default: cem500k)
  --cem-weights PATH               Use a specific checkpoint inside the weights dir

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
EXTRA_ARGS=()

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

DOCKER_IMAGE="cem-fid"
WEIGHTS_DIR="/home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/weights"
RESULTS_DIR="/home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/results"

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/cem_fid"
mkdir -p "$RESULTS_DIR/cem_fid/$CEM_BACKBONE"
mkdir -p "$RESULTS_DIR/normal_fid"

CONTAINER_REAL="/data/real"
CONTAINER_GEN="/data/gen"
CONTAINER_WEIGHTS="/weights"
CONTAINER_RESULTS="/results"

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
CEM_WEIGHTS_CONTAINER=""

if [[ -n "$CEM_WEIGHTS" ]]; then
  CEM_WEIGHTS_REAL=$(normalize_path "$CEM_WEIGHTS")
  if [[ ${CEM_WEIGHTS_REAL} != $WEIGHTS_DIR/* ]]; then
    echo "[ERROR] --cem-weights must reside within $WEIGHTS_DIR" >&2
    exit 4
  fi
  if [[ ! -f "$CEM_WEIGHTS_REAL" ]]; then
    echo "[ERROR] Specified weights file not found: $CEM_WEIGHTS_REAL" >&2
    exit 4
  fi
  CEM_WEIGHTS_CONTAINER="$CONTAINER_WEIGHTS/$(basename "$CEM_WEIGHTS_REAL")"
fi

COMMON_ARGS=("${EXTRA_ARGS[@]}" "--num-workers" "0" "--data-volume" "$GEN_DIR")
CEM_OUTPUT="/results/cem_fid/${CEM_BACKBONE}/cem_fid.json"
NORMAL_OUTPUT="/results/normal_fid/normal_fid.json"
CEM_ARGS=("${COMMON_ARGS[@]}" "--output-json" "$CEM_OUTPUT" "--backbone" "$CEM_BACKBONE")
if [[ -n "$CEM_WEIGHTS_CONTAINER" ]]; then
  CEM_ARGS+=("--weights-path" "$CEM_WEIGHTS_CONTAINER")
fi
NORMAL_ARGS=("${COMMON_ARGS[@]}" "--output-json" "$NORMAL_OUTPUT")

run_in_container() {
  local script_path=$1
  shift
  docker run --rm \
    -v "$REAL_DIR":"$CONTAINER_REAL":ro \
    -v "$GEN_DIR":"$CONTAINER_GEN":ro \
  -v "$WEIGHTS_DIR":"$CONTAINER_WEIGHTS" \
    -v "$RESULTS_DIR":"$CONTAINER_RESULTS" \
    -w /app \
    --entrypoint python \
    "$DOCKER_IMAGE" \
    "$script_path" "$CONTAINER_REAL" "$CONTAINER_GEN" "$@"
}

run_in_container "fid/compute_cem_fid.py" "${CEM_ARGS[@]}"
run_in_container "fid/compute_normal_fid.py" "${NORMAL_ARGS[@]}"
