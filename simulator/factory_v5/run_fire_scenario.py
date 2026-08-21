#!/usr/bin/env python3
"""Run the shared factory_v5 algorithm with one scenario's FDS archives."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIOS_DIR = BASE_DIR / "scenarios"


def load_scenario_metadata(scenario_dir: Path) -> dict[str, Any]:
    metadata_path = scenario_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"scenario metadata not found: {metadata_path}")
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"scenario metadata must be a mapping: {metadata_path}")
    if int(data.get("format_version", 0)) != 1:
        raise ValueError("unsupported or missing scenario format_version")
    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    fire_position = data.get("fire_position_world")
    if fire_position is not None:
        if (
            not isinstance(fire_position, list)
            or len(fire_position) != 2
            or not all(math.isfinite(float(value)) for value in fire_position)
        ):
            raise ValueError("fire_position_world must be finite [x, y] metadata")
    return data


def resolve_scenario_path(
    scenario_dir: Path, metadata: dict[str, Any], field: str
) -> Path:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing scenario path: {field}")
    path = (scenario_dir / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def build_simulation_command(
    scenario_dir: Path, forwarded_args: Sequence[str] = ()
) -> list[str]:
    scenario_dir = scenario_dir.resolve()
    metadata = load_scenario_metadata(scenario_dir)
    temperature_npz = resolve_scenario_path(
        scenario_dir, metadata, "temperature_npz"
    )
    co_npz = resolve_scenario_path(scenario_dir, metadata, "co_npz")
    fds_file = resolve_scenario_path(scenario_dir, metadata, "fds_file")
    result_value = metadata.get("fds_result_dir", ".")
    if not isinstance(result_value, str) or not result_value.strip():
        raise ValueError("fds_result_dir must be a non-empty relative path")
    fds_result_dir = (scenario_dir / result_value).resolve()
    if not fds_result_dir.is_dir():
        raise FileNotFoundError(f"fds_result_dir not found: {fds_result_dir}")
    return [
        sys.executable,
        str(BASE_DIR / "run_partial_costmap_evacuation.py"),
        "--fds-file", str(fds_file),
        "--fds-dir", str(fds_result_dir),
        "--temperature-npz", str(temperature_npz),
        "--co-npz", str(co_npz),
        *forwarded_args,
    ]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        help="scenario ID below scenarios/ or an explicit scenario directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate metadata and print the simulation command without running it",
    )
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    return args, forwarded


def main() -> int:
    args, forwarded = parse_args()
    candidate = Path(args.scenario)
    scenario_dir = candidate if candidate.is_dir() else DEFAULT_SCENARIOS_DIR / candidate
    command = build_simulation_command(scenario_dir, forwarded)
    print("Scenario command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
