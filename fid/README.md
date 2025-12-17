# FID / KID Calculation Tools

This directory contains utility scripts for computing FID (Fréchet Inception Distance) and KID (Kernel Inception Distance). Two main scripts are provided:

- `compute_cem_fid.py`: FID/KID calculation for EM images using CEM pre-trained ResNet50 (CEM500K / CEM1.5M) as the feature extractor.
- `compute_normal_fid.py`: Standard FID/KID calculation using torchvision's ImageNet pre-trained Inception v3.

This document covers dependencies, script overviews, main options, usage examples, and preprocessing differences between the two approaches.

## Setup and Execution Flow

### 1. Pre-trained Weights

- Place CEM500K / CEM1.5M checkpoints under `fid/weights/`.
- See the "Pre-trained weights" section in the repository root `./README.md` for download links and filenames.

### 2. Recommended Execution Methods

- **Local machine**: If Docker is available, we recommend using `run_fid_suite_docker.sh`. Encapsulating dependencies in a pre-built image provides the best reproducibility.
- **Cluster / Non-Docker environments**: Use `run_fid_suite_venv.sh` with a Python venv. Running on GPU nodes significantly speeds up computation (CPU-only execution works but feature extraction takes much longer).

### 3. Environment Setup Helper Script

`fid/setup_fid_env.sh` automates Docker image building and venv creation.

```bash
# Build Docker image only (for local desktop)
./fid/setup_fid_env.sh --mode docker

# Set up venv with CUDA-enabled wheels (for cluster)
./fid/setup_fid_env.sh --mode venv \
  --venv-path /path/to/cem-fid-venv \
  --torch-index https://download.pytorch.org/whl/cu121
```

- Default `--mode auto` builds the Docker image if Docker is found, otherwise creates a venv. Use `--mode both` to prepare both.
- When using venv, specify the same path with `run_fid_suite_venv.sh ... --venv /path/to/cem-fid-venv`.
- The Docker image is tagged `cem-fid` by default. Use `--docker-tag` to specify a different name.

### 4. Execution Summary

1. Place weights under `fid/weights/`.
2. Run `fid/setup_fid_env.sh` to prepare the Docker image or venv.
3. For local execution, use `run_fid_suite_docker.sh REAL_DIR GEN_DIR [OPTIONS] -- [EXTRA_ARGS]`. For clusters, use `run_fid_suite_venv.sh REAL_DIR GEN_DIR --venv /path/to/venv ...`.
4. CEM-FID results are saved to `fid/results/cem_fid/<backbone>/` (e.g., `fid/results/cem_fid/cem500k/`), and normal FID results are saved to `fid/results/normal_fid/`.

> **GPU Recommended**: Both scripts work on CPU-only environments, but GPU execution is tens of times faster for large image sets. In CPU-only environments, reduce `--batch-size` to limit memory usage.

## Prerequisites (Dependencies)

Python packages required by both scripts:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`

If not installed, activate your virtual environment and run:

```bash
pip install torch torchvision numpy scipy tqdm
```

## compute_cem_fid.py (Using CEM ResNet50)

### Overview

`compute_cem_fid.py` uses CEM500K (MoCoV2) or CEM1.5M (SwAV) pre-trained ResNet50 as the feature extractor to compute FID between two EM image directories. KID can also be computed optionally. Grayscale EM images are automatically converted to 3 channels, and the same preprocessing (resize, normalization) used during CEM pre-training is applied to extract 2048-dimensional global average pooling features.

### Basic Usage

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [OPTIONS]
```

- `REAL_DIR`: Directory containing real images
- `GEN_DIR`: Directory containing generated images

### Main Options

| Option | Default | Description |
|---|---:|---|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | CEM pre-trained model to use |
| `--batch-size INT` | `32` | Batch size for feature extraction |
| `--num-workers INT` | `4` | Number of DataLoader worker processes |
| `--device` | Auto (GPU if available) | Inference device |
| `--image-size INT` | `224` | Resize input to this size (same as CEM) |
| `--weights-path PATH` | None | Specify manually downloaded weights |
| `--download-dir PATH` | None | Cache directory for weights (see `TORCH_HOME`) |
| `--output-json PATH` | `cem_fid.json` | Output file (timestamp may be appended) |
| `--compute-kid` | Disabled | Also compute KID (saves features for estimation) |
| `--kid-subset-size INT` | `1000` | Samples per KID subset |
| `--kid-subset-count INT` | `100` | Number of KID subset trials |
| `--seed INT` | `42` | Random seed for KID |

### Output

- Prints FID (and KID mean/stderr if enabled) to console.
- Saves results to the specified `--output-json` as JSON, including FID/KID, backbone, image counts, normalization parameters, UTC timestamp, and input directories.

### Notes

- Pre-trained weights may be automatically downloaded from Zenodo on first run. For offline environments, manually download and specify with `--weights-path`.

## compute_normal_fid.py (Using ImageNet Inception v3)

### Overview

`compute_normal_fid.py` uses torchvision's ImageNet pre-trained Inception v3 to compute FID (and optionally KID) between real and generated image sets. This provides standard Inception-based FID evaluation.

### Basic Usage

```bash
python fid/compute_normal_fid.py REAL_DIR GEN_DIR [OPTIONS]
```

### Main Options

| Option | Default | Description |
|---|---:|---|
| `--batch-size INT` | `32` | Batch size for feature extraction |
| `--num-workers INT` | `4` | Number of DataLoader worker processes |
| `--device` | Auto (GPU if available) | Inference device |
| `--image-size INT` | `299` | Input resolution expected by Inception v3 |
| `--output-json PATH` | `inception_fid.json` | Output file (timestamp may be appended) |
| `--data-volume STR` | None | Environment memo (e.g., host:container mount info) |
| `--compute-kid` | Disabled | Also compute KID (saves features for estimation) |
| `--kid-subset-size INT` | `1000` | Samples per KID subset |
| `--kid-subset-count INT` | `100` | Number of KID subset trials |
| `--seed INT` | `42` | Random seed for KID |

### Output

- Prints FID (and KID mean/stderr if enabled) to console.
- Saves results to the specified `--output-json` as JSON, including FID/KID, backbone name, weight info, image counts, normalization values, UTC timestamp, and input directories.

## Preprocessing Differences

- `compute_cem_fid.py` uses CEM's ResNet50, converting grayscale EM images to 3 channels and applying the same normalization (mean/std) and resolution (224) used during CEM pre-training. Output features are ResNet50's global average pooling (2048 dimensions).
- `compute_normal_fid.py` uses ImageNet pre-trained Inception v3, resizing RGB inputs (grayscale images are converted to RGB) to 299×299 and applying ImageNet normalization. FID/KID is computed using Inception's output features.

## Best Practices

- Specify directories containing only the images to evaluate (scripts recursively search for images).
- Increase `--batch-size` on GPU for faster processing, but watch memory usage.
- When enabling KID, adjust `--kid-subset-size` and `--kid-subset-count` values to balance computation cost.

## Docker Usage Example (CEM ResNet50)

```bash
sudo docker run --rm \
  -v /path/to/real_and_fake:/data \
  -v /path/to/weights:/weights \
  -v /path/to/save/results:/results \
  cem-fid \
  /data/real /data/gen \
  --backbone cem500k \
  --weights-path /weights/cem500k_mocov2_resnet50_200ep.pth.tar \
  --output-json /results/cem_fid.json \
  --data-volume /path/to/real_and_fake:/data
```

## Docker Helper Script (`run_fid_suite_docker.sh`)

The bundled `fid/run_fid_suite_docker.sh` runs both CEM-FID and standard Inception FID on the same dataset sequentially, saving results under `fid/results/`. Key features:

- Run `fid/setup_fid_env.sh --mode docker` beforehand to build the `cem-fid` image.
- Specify `REAL_DIR` and `GEN_DIR` on the host; they are automatically mounted to `/data/real` and `/data/gen`.
- Use `--cem-backbone {cem500k|cem1.5m}` to switch between MoCoV2 (CEM500K) and SwAV (CEM1.5M). Specify multiple times to run CEM-FID with all specified backbones sequentially.
- Checkpoints in `fid/weights/` are auto-detected (or explicitly specify with `--cem-weights`).
- Arguments after `--` are forwarded to both Python scripts (e.g., `--batch-size 64`).

Example running CEM-FID with SwAV backbone:

```bash
./fid/run_fid_suite_docker.sh /path/to/real /path/to/gen \
  --cem-backbone cem1.5m \
  --cem-weights /path/to/weights/cem1.5m_swav_resnet50_200ep_balanced.pth.tar \
  -- --batch-size 64
```

CEM-FID is computed with the SwAV backbone, followed by standard Inception FID on the same data. Results are saved to `fid/results/cem_fid/cem1.5m/` and `fid/results/normal_fid/` with timestamps, making comparison and log management easy.

To evaluate with both MoCoV2 and SwAV, specify `--cem-backbone` multiple times:

```bash
./fid/run_fid_suite_docker.sh /path/to/real /path/to/gen \
  --cem-backbone cem500k \
  --cem-backbone cem1.5m \
  -- --batch-size 32
```

This runs CEM-FID twice (cem500k → cem1.5m), saving results to `fid/results/cem_fid/cem500k/` and `fid/results/cem_fid/cem1.5m/`. Standard Inception FID runs once, saving to `fid/results/normal_fid/`.

### Batch Processing Helper (`run_fid_suite_batch.py`)

For batch evaluation of multiple datasets with the same settings, use `fid/run_fid_suite_batch.py` which reads a JSON manifest. Jobs in the `jobs` array are executed sequentially via `run_fid_suite_docker.sh`.

```json
{
  "jobs": [
    {
      "name": "example-job",
      "real_dir": "/abs/path/to/real_images",
      "gen_dir": "/abs/path/to/generated_images",
      "cem_backbones": ["cem500k", "cem1.5m"],
      "extra_args": ["--batch-size", "256"]
    }
  ]
}
```

Prepare a file like `fid/batch_jobs.example.json` and run:

```bash
python fid/run_fid_suite_batch.py fid/batch_jobs.example.json
```

To apply global options to all jobs, add them after `--`:

```bash
python fid/run_fid_suite_batch.py fid/batch_jobs.example.json -- --batch-size 64
```

- `--script` lets you substitute the suite script (default: `run_fid_suite_docker.sh`).
- `--jobs-base` resolves relative `real_dir` / `gen_dir` paths against a base path.
- Each job entry can configure:
  - `cem_backbones`: List of backbones to use (`["cem500k"]`, `["cem1.5m"]`, `["cem500k", "cem1.5m"]`, etc.)
  - `cem_weights`: Custom weights file path (only for single backbone)
  - `script_args`: Additional CLI options for the suite script (inserted before `--`)
  - `extra_args`: Options passed to Python scripts (passed after `--`)
- Control options like `--stop-on-error`, `--dry-run`, `--json-log`, and `--quiet` are available. See `python fid/run_fid_suite_batch.py --help` for details.

## venv Helper Script (`run_fid_suite_venv.sh`)

For clusters without Docker, prepare a Python venv with dependencies using `fid/setup_fid_env.sh --mode venv --venv-path /path/to/venv`, then run:

```bash
./fid/run_fid_suite_venv.sh REAL_DIR GEN_DIR \
  --venv /path/to/venv \
  --cem-backbone cem1.5m \
  -- --batch-size 64 --device cuda
```

> **Note**: Unlike the Docker version, `run_fid_suite_venv.sh` only supports specifying `--cem-backbone` **once** (for evaluating multiple backbones, run the script multiple times or use the Docker version / batch processing helper).

- If `--venv` is omitted, `fid/../venv` or `fid/venv` is auto-detected. Specify the path explicitly when sharing among multiple users.
- On GPU nodes, adjust `--device cuda` (auto-detected by default) and `--batch-size` for your environment. CPU-only nodes work but are very slow.

## Additional Notes

- For offline environments, manually download pre-trained weights and pass them with `--weights-path`.
- Each script saves execution metadata (timestamp, input directories, options used, etc.) to JSON for experiment logging.

## PNG Integrity Checker (`fid/utils/check_png_integrity.py`)

A lightweight utility for checking PNG file corruption before processing large sets of generated images.

```bash
source .venv/bin/activate
pip install pillow tqdm
python fid/utils/check_png_integrity.py /path/to/images
```

- Recursively scans PNG files under the specified directory and reports any files that fail to open (raise exceptions in `PIL.Image`).
- If no corrupted files are found, displays "All ... PNG file(s) opened successfully." with the count.
- Progress bar is shown automatically if `tqdm` is installed (use `--no-progress` to disable).
- Install Pillow (`pip install pillow`) and `tqdm` if not already present.

---

Script files:

- `fid/compute_cem_fid.py`
- `fid/compute_normal_fid.py`

See the command-line argument descriptions in each source file for details.
