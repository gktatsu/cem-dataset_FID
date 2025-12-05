#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: setup_fid_env.sh [OPTIONS]

Automates the recommended environment setup for running FID/KID evaluation.
By default it builds the Docker image when Docker is available; otherwise it
creates/updates a Python venv with the required dependencies.

Options:
  --mode {auto|docker|venv|both}  What to configure (default: auto)
  --docker-tag TAG                Docker image tag to build (default: cem-fid)
  --venv-path PATH                Path to create/update the venv (default: <repo>/venv)
  --python PYTHON                Python executable for venv creation (default: python3)
  --torch-index URL              Optional pip index/extra-index for torch wheels
  --help                         Show this help and exit

Examples:
  # Build Docker image only
  ./fid/setup_fid_env.sh --mode docker

  # Create venv in ./venv with CUDA wheels
  ./fid/setup_fid_env.sh --mode venv --torch-index https://download.pytorch.org/whl/cu121

  # Force both Docker build and venv setup
  ./fid/setup_fid_env.sh --mode both --venv-path /scratch/$USER/cem-fid-venv
USAGE
}

MODE="auto"
DOCKER_TAG="cem-fid"
PYTHON_BIN="python3"
TORCH_INDEX=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_VENV="$REPO_ROOT/venv"
VENV_PATH="$DEFAULT_VENV"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE=${2:-}
      if [[ -z "$MODE" ]]; then
        echo "[ERROR] --mode requires an argument" >&2
        exit 1
      fi
      shift 2
      ;;
    --docker-tag)
      DOCKER_TAG=${2:-}
      if [[ -z "$DOCKER_TAG" ]]; then
        echo "[ERROR] --docker-tag requires an argument" >&2
        exit 1
      fi
      shift 2
      ;;
    --venv-path)
      VENV_PATH=${2:-}
      if [[ -z "$VENV_PATH" ]]; then
        echo "[ERROR] --venv-path requires an argument" >&2
        exit 1
      fi
      shift 2
      ;;
    --python)
      PYTHON_BIN=${2:-}
      if [[ -z "$PYTHON_BIN" ]]; then
        echo "[ERROR] --python requires an argument" >&2
        exit 1
      fi
      shift 2
      ;;
    --torch-index)
      TORCH_INDEX=${2:-}
      if [[ -z "$TORCH_INDEX" ]]; then
        echo "[ERROR] --torch-index requires an argument" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac

done

ensure_dirs() {
  mkdir -p "$SCRIPT_DIR/weights"
  mkdir -p "$SCRIPT_DIR/results/cem_fid"
  mkdir -p "$SCRIPT_DIR/results/normal_fid"
}

build_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or not on PATH" >&2
    exit 2
  fi
  echo "[INFO] Building Docker image '$DOCKER_TAG' using $REPO_ROOT/Dockerfile" >&2
  docker build -t "$DOCKER_TAG" -f "$REPO_ROOT/Dockerfile" "$REPO_ROOT"
}

setup_venv() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    exit 3
  fi

  echo "[INFO] Creating/updating venv at $VENV_PATH" >&2
  "$PYTHON_BIN" -m venv "$VENV_PATH"

  PIP_BIN="$VENV_PATH/bin/pip"
  if [[ ! -x "$PIP_BIN" ]]; then
    echo "[ERROR] pip not found in venv ($PIP_BIN)" >&2
    exit 4
  fi

  "$PIP_BIN" install --upgrade pip wheel

  TORCH_ARGS=(install --no-cache-dir torch torchvision)
  if [[ -n "$TORCH_INDEX" ]]; then
    TORCH_ARGS+=(--index-url "$TORCH_INDEX")
  fi
  "$PIP_BIN" "${TORCH_ARGS[@]}"

  "$PIP_BIN" install --no-cache-dir numpy scipy pillow tqdm
}

should_run_docker=false
should_run_venv=false

case "$MODE" in
  auto)
    if command -v docker >/dev/null 2>&1; then
      should_run_docker=true
    else
      should_run_venv=true
    fi
    ;;
  docker)
    should_run_docker=true
    ;;
  venv)
    should_run_venv=true
    ;;
  both)
    should_run_docker=true
    should_run_venv=true
    ;;
  *)
    echo "[ERROR] Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

ensure_dirs

if [[ "$should_run_docker" == true ]]; then
  build_docker
fi

if [[ "$should_run_venv" == true ]]; then
  setup_venv
fi

if [[ "$should_run_docker" == false && "$should_run_venv" == false ]]; then
  echo "[WARN] Nothing to do (check --mode)." >&2
fi

docker_status="skipped"
if [[ "$should_run_docker" == true ]]; then
  docker_status="built"
fi

venv_status="skipped"
if [[ "$should_run_venv" == true ]]; then
  venv_status="configured at $VENV_PATH"
fi

cat <<SUMMARY
Setup completed.
- Docker: $docker_status
- venv:   $venv_status

Weights directory : $SCRIPT_DIR/weights
Results directory : $SCRIPT_DIR/results

Next steps:
  1. Download pre-trained weights as described in ./README.md and place them under fid/weights.
  2. Use run_fid_suite_docker.sh (local) or run_fid_suite_venv.sh (cluster) to execute evaluations.
SUMMARY
