# Fire scenarios

`factory_v5`의 Python 알고리즘과 건물 geometry는 상위 디렉터리에서 한 번만
관리한다. 이 폴더에는 화재 위치별 FDS 결과와 runtime NPZ만 분리해 둔다.

각 시나리오는 다음 구조를 사용한다.

```text
scenarios/<scenario_id>/
├── metadata.yaml
├── fire.inc                 # 선택 사항: 해당 실험의 화재 정의
├── fds_result/              # 원본 FDS 결과(대용량, Git 제외)
└── processed/
    ├── fds_temperature_3d_timeseries.npz
    └── fds_co_2d_timeseries.npz
```

`fire_position_world`는 실험을 식별하고 사후 오차를 평가하기 위한 metadata이며,
로봇의 추정·Costmap·경로 선택 입력으로 직접 사용하지 않는다.

새 시나리오는 `template/`을 복사해 만들고 FDS `CHID`는 서로 다르게 지정한다.
변환과 실행 방법은 상위 `README.md`의 **Multiple fire scenarios** 절을 따른다.
