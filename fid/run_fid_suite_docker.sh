#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_fid_suite_docker.sh REAL_DIR GEN_DIR [--] [EXTRA_ARGS...]

Runs compute_cem_fid.py and compute_normal_fid.py inside a fixed Docker
container environment using the host directories specified below.

Constants:
  Docker image : cem-fid
  Weights dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/weights
  Results dir  : /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/results

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
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    --)
      shift
      EXTRA_ARGS=("$@")
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
fi

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
mkdir -p "$RESULTS_DIR/normal_fid"

CONTAINER_REAL="/data/real"
CONTAINER_GEN="/data/gen"
CONTAINER_WEIGHTS="/weights"
CONTAINER_RESULTS="/results"

COMMON_ARGS=("${EXTRA_ARGS[@]}" "--num-workers" "0" "--data-volume" "$GEN_DIR")
CEM_OUTPUT="/results/cem_fid/cem_fid.json"
NORMAL_OUTPUT="/results/normal_fid/normal_fid.json"
CEM_ARGS=("${COMMON_ARGS[@]}" "--output-json" "$CEM_OUTPUT")
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
