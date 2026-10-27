#!/usr/bin/env python3
"""Submit all configured experiments as a Slurm array, or retry array tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from run_experiment import CONFIG_DIR, PROJECT_ROOT, eligible_indices, load_config


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_indices(value: str, count: int) -> list[int]:
    try:
        indices = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("retry indices must be comma-separated integers") from error
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= count for i in indices):
        raise ValueError(f"retry indices must be unique and between 0 and {count - 1}")
    return sorted(indices)


def create_manifest(savedir: Path) -> Path:
    paths = sorted(CONFIG_DIR.glob("*.json"))
    if not paths:
        raise ValueError(f"No experiment configurations found in {CONFIG_DIR}")
    submission_root = savedir / "slurm-submissions"
    submission_root.mkdir(parents=True, exist_ok=True)
    submission_dir = Path(tempfile.mkdtemp(prefix="sweep-", dir=submission_root))
    snapshots = submission_dir / "configs"
    snapshots.mkdir()
    (submission_dir / "logs").mkdir()
    tasks = []
    for path in paths:
        snapshot = snapshots / path.name
        shutil.copyfile(path, snapshot)
        config = load_config(snapshot)
        for index in eligible_indices(config):
            tasks.append({"config": str(snapshot), "name": config["name"], "index": index})
    if not tasks:
        raise ValueError("No eligible experiment runs found")
    manifest = submission_dir / "runs.json"
    manifest.write_text(json.dumps({"savedir": str(savedir), "tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_task(manifest_path: Path, task_index: int, python: str, savedir: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task_index < 0 or task_index >= len(manifest["tasks"]):
        raise ValueError(f"Array task {task_index} is not in {manifest_path}")
    task = manifest["tasks"][task_index]
    print(f"Array task {task_index}: {task['name']} (original index {task['index']})", flush=True)
    command = [python, str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
               task["config"], "--index", str(task["index"]), "--savedir", savedir,
               "--python", python]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=positive_int, default=20)
    parser.add_argument("--memory", default="30G", help="memory per job, e.g. 30G")
    parser.add_argument("--time", default="24:00:00", help="Slurm time limit")
    parser.add_argument("--savedir", default=None, help="experiment output root (default: ./results; retries reuse the original)")
    parser.add_argument("--sbatch-option", action="append", default=[], help="additional sbatch option; use --sbatch-option=--partition=NAME")
    parser.add_argument("--retry", type=Path, help="reuse an existing runs.json")
    parser.add_argument("--indices", help="comma-separated Slurm task indices to retry")
    parser.add_argument("--dry-run", action="store_true", help="prepare the run list and print sbatch without submitting")
    parser.add_argument("--run-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--python", default=sys.executable, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_task is not None:
        if args.task_index is None:
            parser.error("--run-task requires --task-index")
        if args.savedir is None:
            parser.error("--run-task requires --savedir")
        run_task(args.run_task, args.task_index, args.python, str(Path(args.savedir).resolve()))
        return
    if (args.retry is None) != (args.indices is None):
        parser.error("--retry and --indices must be supplied together")
    if not os.environ.get("CONDA_PREFIX"):
        parser.error("activate the project Conda environment before submitting")
    blocked = ("--array", "--output", "--error", "--chdir", "--wrap")
    if any(option.split("=", 1)[0] in blocked for option in args.sbatch_option):
        parser.error("--sbatch-option cannot override the array, log paths, working directory, or script")
    savedir = Path(args.savedir).resolve() if args.savedir else (PROJECT_ROOT / "results")
    manifest_path = args.retry.resolve() if args.retry else create_manifest(savedir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.retry and args.savedir is None:
        savedir = Path(manifest["savedir"])
    count = len(manifest["tasks"])
    if not count:
        parser.error("manifest contains no tasks")
    try:
        indices = parse_indices(args.indices, count) if args.indices else None
    except ValueError as error:
        parser.error(str(error))
    array = ",".join(map(str, indices)) if indices is not None else f"0-{count - 1}"
    logs = manifest_path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = PROJECT_ROOT / "scripts" / "slurm_experiments.sh"
    command = ["sbatch", "--parsable", "--job-name=icl-sweep", f"--array={array}%{args.concurrency}",
               "--gres=gpu:1", "--cpus-per-task=1", f"--mem={args.memory}", f"--time={args.time}",
               f"--chdir={PROJECT_ROOT}", f"--output={logs}/%A_%a.out", f"--error={logs}/%A_%a.err",
               *args.sbatch_option, str(worker), str(manifest_path), sys.executable, str(savedir), str(PROJECT_ROOT)]
    print(f"Run list: {manifest_path} ({count} total tasks)", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(f"Submitted Slurm array: {result.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
