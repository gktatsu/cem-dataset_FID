#!/usr/bin/env python3
"""Batch runner for run_fid_suite_docker.sh.

This helper reads a JSON manifest that lists multiple REAL/GEN directory pairs
with optional per-job options and executes run_fid_suite_docker.sh for each job
sequentially. See fid/batch_jobs.example.json for the manifest structure.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE_SCRIPT = REPO_ROOT / "run_fid_suite_docker.sh"


def parse_args() -> argparse.Namespace:
    epilog = (
        "Example:\n"
        "  ./fid/run_fid_suite_batch.py fid/batch_jobs.example.json -- \\\n"
        "    --batch-size 64\n"
        "This forwards '--batch-size 64' to both compute_cem_fid.py and\n"
        "compute_normal_fid.py via run_fid_suite_docker.sh for every job."
    )
    parser = argparse.ArgumentParser(
        description="Batch runner for multiple FID suite evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "jobs_file",
        help=(
            "Path to a JSON file containing a list of jobs or an object with a"
            " 'jobs' list"
        ),
    )
    parser.add_argument(
        "--script",
        default=str(DEFAULT_SUITE_SCRIPT),
        help=(
            "Path to run_fid_suite_docker.sh (default: "
            f"{DEFAULT_SUITE_SCRIPT})"
        ),
    )
    parser.add_argument(
        "--jobs-base",
        help=(
            "If set, resolve relative real/gen directories against this base"
            " path"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run but do not execute them",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch when a job fails (default: keep going)",
    )
    parser.add_argument(
        "--json-log",
        help="Append newline-delimited JSON status objects to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print job summaries (suppresses per-command echo)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended after '--' for every run",
    )
    return parser.parse_args()


@dataclass
class Job:
    name: str
    real_dir: str
    gen_dir: str
    backbones: List[str]
    cem_weights: str | None
    script_args: List[str]
    extra_args: List[str]


def load_jobs(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        jobs = data.get("jobs")
        if jobs is None:
            raise ValueError(
                "JSON object must contain a 'jobs' key with a list"
            )
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError(
            "Jobs manifest must be either a list or an object with 'jobs' list"
        )
    if not isinstance(jobs, list):
        raise ValueError("'jobs' must be a list")
    return jobs


def ensure_list(value, *, field: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        result = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"All items in '{field}' must be strings")
            result.append(item)
        return result
    raise ValueError(f"Field '{field}' must be a string or list of strings")


def normalize_backbones(values: Sequence[str]) -> List[str]:
    allowed = {"cem500k", "cem1.5m"}
    seen = set()
    result: List[str] = []
    for raw in values or ["cem500k"]:
        if raw not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"Unsupported backbone '{raw}'. Expected one of {allowed_list}"
            )
        if raw not in seen:
            seen.add(raw)
            result.append(raw)
    return result


def resolve_path(base: Path | None, candidate: str) -> str:
    path = Path(candidate)
    if not path.is_absolute() and base is not None:
        path = (base / path).resolve()
    return str(path)


def build_job(obj: dict, base: Path | None) -> Job:
    required = ["real_dir", "gen_dir"]
    for field in required:
        if field not in obj:
            raise ValueError(f"Job is missing required field '{field}'")
    name = obj.get("name") or f"{obj['real_dir']} -> {obj['gen_dir']}"
    backbones = normalize_backbones(
        ensure_list(obj.get("cem_backbones"), field="cem_backbones")
    )
    cem_weights = obj.get("cem_weights")
    if cem_weights is not None:
        if len(backbones) != 1:
            message = (
                f"Job '{name}' specifies cem_weights but also multiple "
                "backbones. Provide separate jobs when custom weights differ."
            )
            raise ValueError(message)
        cem_weights = resolve_path(base, cem_weights)
    script_args = ensure_list(obj.get("script_args"), field="script_args")
    extra_args = ensure_list(obj.get("extra_args"), field="extra_args")
    real_dir = resolve_path(base, obj["real_dir"])
    gen_dir = resolve_path(base, obj["gen_dir"])
    return Job(
        name=name,
        real_dir=real_dir,
        gen_dir=gen_dir,
        backbones=backbones,
        cem_weights=cem_weights,
        script_args=script_args,
        extra_args=extra_args,
    )


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def append_json_log(log_path: Path, payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    args = parse_args()
    jobs_file = Path(args.jobs_file).expanduser().resolve()
    jobs_base = (
        Path(args.jobs_base).expanduser().resolve() if args.jobs_base else None
    )
    suite_script = Path(args.script).expanduser().resolve()
    if not suite_script.exists():
        print(
            f"[ERROR] run_fid_suite script not found: {suite_script}",
            file=sys.stderr,
        )
        return 2

    try:
        raw_jobs = load_jobs(jobs_file)
        jobs = [build_job(obj, jobs_base) for obj in raw_jobs]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to load jobs: {exc}", file=sys.stderr)
        return 2

    if not jobs:
        print("[WARN] No jobs found in manifest. Nothing to do.")
        return 0

    global_extra = list(args.extra_args or [])
    log_path = (
        Path(args.json_log).expanduser().resolve() if args.json_log else None
    )

    successes = 0
    failures = 0

    for idx, job in enumerate(jobs, start=1):
        cmd: List[str] = [str(suite_script), job.real_dir, job.gen_dir]
        for backbone in job.backbones:
            cmd.extend(["--cem-backbone", backbone])
        if job.cem_weights:
            cmd.extend(["--cem-weights", job.cem_weights])
        if job.script_args:
            cmd.extend(job.script_args)
        combined_extra = job.extra_args + global_extra
        if combined_extra:
            cmd.append("--")
            cmd.extend(combined_extra)

        heading = f"[JOB {idx}/{len(jobs)}] {job.name}"
        if not args.quiet:
            print("=" * len(heading))
            print(heading)
            print("=" * len(heading))
        else:
            print(heading)
        print(quote_cmd(cmd))

        status = {
            "name": job.name,
            "index": idx,
            "total": len(jobs),
            "real_dir": job.real_dir,
            "gen_dir": job.gen_dir,
            "backbones": job.backbones,
            "command": cmd,
        }

        if args.dry_run:
            status["returncode"] = None
            status["status"] = "skipped"
            if log_path:
                append_json_log(log_path, status)
            continue

        try:
            completed = subprocess.run(cmd, check=False)
            rc = completed.returncode
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user", file=sys.stderr)
            return 130

        status["returncode"] = rc
        if rc == 0:
            successes += 1
            status["status"] = "ok"
            print("-> SUCCESS")
        else:
            failures += 1
            status["status"] = "failed"
            print(f"-> FAILED (return code {rc})", file=sys.stderr)
            if args.stop_on_error:
                if log_path:
                    append_json_log(log_path, status)
                break
        if log_path:
            append_json_log(log_path, status)

    print(f"Completed batch: {successes} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
