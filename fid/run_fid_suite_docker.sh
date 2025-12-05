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
  --cem-backbone {cem500k|cem1.5m}  Select a CEM backbone (repeatable, default: cem500k)
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

CEM_BACKBONES=()
CEM_WEIGHTS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cem-backbone)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --cem-backbone requires an argument" >&2
        exit 1
      fi
      CEM_BACKBONES+=("$2")
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

if [[ ${#CEM_BACKBONES[@]} -eq 0 ]]; then
  CEM_BACKBONES=("cem500k")
fi

declare -A _SEEN_BACKBONES=()
declare -a CEM_BACKBONES_DEDUP=()
for backbone in "${CEM_BACKBONES[@]}"; do
  case "$backbone" in
    cem500k|cem1.5m)
      if [[ -z "${_SEEN_BACKBONES[$backbone]+x}" ]]; then
        _SEEN_BACKBONES[$backbone]=1
        CEM_BACKBONES_DEDUP+=("$backbone")
      fi
      ;;
    *)
      echo "[ERROR] --cem-backbone must be cem500k or cem1.5m (got $backbone)" >&2
      exit 1
      ;;
  esac
done
unset _SEEN_BACKBONES
CEM_BACKBONES=("${CEM_BACKBONES_DEDUP[@]}")
unset CEM_BACKBONES_DEDUP

if [[ -n "$CEM_WEIGHTS" && ${#CEM_BACKBONES[@]} -gt 1 ]]; then
  echo "[ERROR] --cem-weights cannot be combined with multiple --cem-backbone values." >&2
  exit 1
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
for backbone in "${CEM_BACKBONES[@]}"; do
  mkdir -p "$RESULTS_DIR/cem_fid/$backbone"
done
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

COMMON_ARGS=("${EXTRA_ARGS[@]}" "--num-workers" "0" "--data-volume" "$GEN_DIR")
NORMAL_OUTPUT="/results/normal_fid/normal_fid.json"
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

for backbone in "${CEM_BACKBONES[@]}"; do
  cem_weights_real=""
  cem_weights_container=""
  weights_candidate="$CEM_WEIGHTS"

  if [[ -z "$weights_candidate" ]]; then
    default_name=$(select_default_weights "$backbone")
    if [[ -n "$default_name" && -f "$WEIGHTS_DIR/$default_name" ]]; then
      weights_candidate="$WEIGHTS_DIR/$default_name"
    fi
  fi

  if [[ -n "$weights_candidate" ]]; then
    cem_weights_real=$(normalize_path "$weights_candidate")
    if [[ ${cem_weights_real} != $WEIGHTS_DIR/* ]]; then
      echo "[ERROR] --cem-weights must reside within $WEIGHTS_DIR" >&2
      exit 4
    fi
    if [[ ! -f "$cem_weights_real" ]]; then
      echo "[ERROR] Specified weights file not found: $cem_weights_real" >&2
      exit 4
    fi
    cem_weights_container="$CONTAINER_WEIGHTS/$(basename "$cem_weights_real")"
  fi

  cem_output="/results/cem_fid/${backbone}/cem_fid.json"
  cem_args=("${COMMON_ARGS[@]}" "--output-json" "$cem_output" "--backbone" "$backbone")
  if [[ -n "$cem_weights_container" ]]; then
    cem_args+=("--weights-path" "$cem_weights_container")
  fi

  run_in_container "fid/compute_cem_fid.py" "${cem_args[@]}"
done

run_in_container "fid/compute_normal_fid.py" "${NORMAL_ARGS[@]}"
