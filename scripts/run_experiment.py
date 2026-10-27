#!/usr/bin/env python3
"""Launch reproducible experiments declared in ``configs/experiments``.

The published configurations are deliberately data-only JSON files.  This
launcher converts a Slurm-array index into the matching command without
submitting anything itself.  Use ``--dry-run`` to inspect a command first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"name", "entrypoint", "arguments"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    return config


def configurations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a product sweep, with its first axis varying fastest."""
    sweep = config.get("sweep", {})
    if "runs" in sweep:
        runs = sweep["runs"]
        if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
            raise ValueError("sweep.runs must be a list of argument dictionaries.")
        return runs
    axes = sweep.get("axes", [])
    if not axes:
        return [{}]
    names = [axis["argument"] for axis in axes]
    values = []
    for axis in axes:
        if "values" in axis:
            values.append(axis["values"])
        elif "range" in axis:
            bounds = axis["range"]
            values.append(list(range(bounds["start"], bounds["stop"], bounds.get("step", 1))))
        else:
            raise ValueError(f"Sweep axis {axis['argument']!r} needs 'values' or 'range'.")
    if any(not value for value in values):
        raise ValueError("Every sweep axis must contain at least one value.")
    # itertools.product changes its right-most input fastest; reverse twice so
    # the first documented axis matches the historical Slurm decoding.
    return [
        dict(zip(names, combination[::-1]))
        for combination in itertools.product(*values[::-1])
    ]


def flag_and_value(argument: str, value: Any) -> list[str]:
    flag = f"--{argument}"
    if isinstance(value, bool):
        return [flag if value else f"--no-{argument}"]
    if value is None:
        return []
    return [flag, str(value)]


def eligible_indices(config: dict[str, Any]) -> list[int]:
    """Select valid array indices while retaining the original index values."""
    runs = configurations(config)
    selection = config.get("sweep", {}).get("allowed_head_feature_products")
    if selection is None:
        return list(range(len(runs)))
    heads = selection["heads"]
    features = selection["features"]
    products = set(selection["products"])
    return [
        index for index, run in enumerate(runs)
        if heads[run["SLURM_ARRAY_TASK_ID"] % len(heads)]
        * features[(run["SLURM_ARRAY_TASK_ID"] // len(heads)) % len(features)]
        in products
    ]


def build_command(
    config: dict[str, Any], index: int, python: str, savedir: str | None
) -> list[str]:
    runs = configurations(config)
    if not 0 <= index < len(runs):
        raise IndexError(
            f"index {index} is outside the configured range 0-{len(runs) - 1}."
        )
    arguments = dict(config["arguments"])
    arguments.update(runs[index])
    if savedir is not None:
        arguments["savedir"] = savedir

    entrypoint = PROJECT_ROOT / config["entrypoint"]
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Entrypoint does not exist: {entrypoint}")
    command = [python, "-u", str(entrypoint)]
    for argument, value in arguments.items():
        command.extend(flag_and_value(argument, value))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", help="JSON experiment configuration")
    parser.add_argument("--all", action="store_true", help="run all eligible indices of all configurations sequentially")
    parser.add_argument("--index", type=int, default=None, help="zero-based sweep index")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--savedir",
        default=str(PROJECT_ROOT / "results"),
        help="override the configuration output root (recommended for new runs)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, but do not run")
    parser.add_argument(
        "--print-array-size", action="store_true", help="print the number of sweep jobs and exit"
    )
    parsed = parser.parse_args()

    if parsed.all:
        if parsed.config is not None or parsed.index is not None or parsed.print_array_size:
            parser.error("--all cannot be combined with a config, --index, or --print-array-size")
        paths = sorted(CONFIG_DIR.glob("*.json"))
        if not paths:
            parser.error(f"No experiment configs found in {CONFIG_DIR}")
        tasks = []
        for path in paths:
            config = load_config(path)
            tasks.extend((config, index) for index in eligible_indices(config))
        print(f"Found {len(tasks)} eligible runs across {len(paths)} configurations.", flush=True)
        failures = []
        for number, (config, index) in enumerate(tasks, start=1):
            print(f"[{number}/{len(tasks)}] {config['name']} index {index}", flush=True)
            command = build_command(config, index, parsed.python, parsed.savedir)
            print(shlex.join(command), flush=True)
            if parsed.dry_run:
                continue
            environment = os.environ.copy()
            environment.update({str(key): str(value) for key, value in config.get("environment", {}).items()})
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
            except (subprocess.CalledProcessError, OSError) as error:
                failures.append((config["name"], index, error))
                print(f"Run failed: {error}", file=sys.stderr, flush=True)
        if failures:
            print(f"{len(failures)} run(s) failed:", file=sys.stderr)
            for name, index, error in failures:
                print(f"  {name} index {index}: {error}", file=sys.stderr)
            raise SystemExit(1)
        return

    if parsed.config is None:
        parser.error("provide a config or use --all")

    config_path = parsed.config if parsed.config.is_absolute() else PROJECT_ROOT / parsed.config
    config = load_config(config_path)
    runs = configurations(config)
    if parsed.print_array_size:
        print(len(runs))
        return
    command = build_command(config, parsed.index if parsed.index is not None else 0, parsed.python, parsed.savedir)
    print(shlex.join(command), flush=True)
    if parsed.dry_run:
        return
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in config.get("environment", {}).items()})
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
