# Repository Guidelines

## Project Structure & Module Organization

This repository simulates a fire-evacuation robot in a 20 m × 30 m FDS factory. `factory_v1.fds` is the authoritative map and obstacle definition. Pygame entry points live at the repository root: use `pygame_factory_map.py` for the basic A* map and `pygame_map_matplotlib_thermal_viewer.py` for the integrated robot, thermal-camera, and MQ-135 view. Path planning is split between `mapping/grid_map.py` and `planner/a_star.py`. Sensor models are under `sensors/`; detection helpers are in `human_detection_sim.py`. FDS outputs, CSV files, and `processed/*.npz` are simulation assets, not source code. Existing `test_*.py` files are executable diagnostic scripts.

## Build, Test, and Development Commands

There is no build step or checked-in dependency manifest. Use a Python environment containing Pygame, NumPy, Matplotlib, and `fdsreader`.

```bash
python pygame_factory_map.py
python pygame_map_matplotlib_thermal_viewer.py
python -m py_compile pygame_factory_map.py mapping/*.py planner/*.py sensors/*.py
python test_fds_thermal.py
```

The first command runs the static A* demonstration. The second loads the processed thermal volume and FDS CO slice; press Enter to start. `py_compile` is the minimum syntax check before submitting changes. Run relevant diagnostic scripts directly when modifying thermal logic.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for simulation constants. Keep world coordinates in metres and radians internally. Convert to Pygame pixels only in `WorldTransform`. Add type hints and short docstrings to reusable mapping, planning, and sensor APIs. No formatter or linter is currently configured; avoid drive-by formatting of unrelated files.

## Testing Guidelines

No coverage threshold or automated test framework is configured. Name new tests `test_<feature>.py`. For planner changes, verify that paths begin/end correctly, contain no occupied cells, reject diagonal corner cutting, and reach the final world position using frame `dt`. GUI changes should include a manual run and, when practical, a headless SDL smoke test.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so no repository-specific convention can be inferred. Use short imperative subjects such as `Add exit-aware A* planning`. Keep commits focused. Pull requests should describe behavior changes, list commands run, identify modified FDS/data assumptions, and include screenshots for map or sensor-display changes. Link the relevant issue when one exists.

## Data and Configuration

Do not commit regenerated FDS outputs or large NPZ/CSV artifacts unless the change explicitly requires them. Preserve sensor code and source data when changing only navigation behavior.

# Project Overview

이 프로젝트는 FDS 화재 환경 데이터를 이용한 화재 대피 로봇
Python 시뮬레이션이다.

## Directory responsibilities

- sensors/: 가상 열화상 카메라와 가스센서
- robot/: 로봇 상태와 이동
- mapping/: 장애물 맵과 동적 costmap
- planner/: A* 및 경로 계획
- processed/: FDS 전처리 결과

## Working rules

- 기존 열화상 카메라와 MQ-135 구현을 임의로 삭제하지 않는다.
- FDS 원본 파일은 명시적인 요청 없이 변경하지 않는다.
- 좌표는 월드 좌표 단위 m와 격자 좌표를 명확히 구분한다.
- 배열은 map[y, x], 경로 노드는 (x, y) 순서를 사용한다.
- Pygame 화면의 y축 반전을 명시적으로 처리한다.
- 관련 없는 코드 전체를 리팩터링하지 않는다.
- 수정 후 실행 방법과 변경 파일을 보고한다.
