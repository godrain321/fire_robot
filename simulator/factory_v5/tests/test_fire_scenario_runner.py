from pathlib import Path

import pytest
import yaml

from run_fire_scenario import build_simulation_command, load_scenario_metadata


def _scenario(tmp_path: Path) -> Path:
    scenario = tmp_path / "case_01"
    processed = scenario / "processed"
    processed.mkdir(parents=True)
    (processed / "temperature.npz").touch()
    (processed / "co.npz").touch()
    (scenario / "case.fds").write_text("&HEAD CHID='case' /\n", encoding="utf-8")
    (scenario / "metadata.yaml").write_text(
        yaml.safe_dump({
            "format_version": 1,
            "scenario_id": "case_01",
            "fds_file": "case.fds",
            "temperature_npz": "processed/temperature.npz",
            "co_npz": "processed/co.npz",
            "fire_position_world": [1.0, 2.0],
        }),
        encoding="utf-8",
    )
    return scenario


def test_build_command_uses_scenario_archives(tmp_path: Path):
    scenario = _scenario(tmp_path)
    command = build_simulation_command(scenario, ["--headless"])
    assert command[-1] == "--headless"
    assert str((scenario / "processed/temperature.npz").resolve()) in command
    assert str((scenario / "processed/co.npz").resolve()) in command
    assert str((scenario / "case.fds").resolve()) in command


def test_missing_archive_is_rejected(tmp_path: Path):
    scenario = _scenario(tmp_path)
    (scenario / "processed/co.npz").unlink()
    with pytest.raises(FileNotFoundError, match="co_npz"):
        build_simulation_command(scenario)


def test_invalid_fire_position_metadata_is_rejected(tmp_path: Path):
    scenario = _scenario(tmp_path)
    metadata_path = scenario / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["fire_position_world"] = [float("nan"), 2.0]
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fire_position_world"):
        load_scenario_metadata(scenario)
