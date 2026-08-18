# Fire Robot Simulation

이 저장소는 FDS 화재 환경에서 로봇 센서 관측, 부분 Costmap, 경로 계획,
요구조자 탐색·동행 대피를 검증하기 위한 시뮬레이션 프로젝트다. 현재 최종 통합
시뮬레이션은 `simulator/factory_v5`이다.

## 1. Factory 버전별 차이

| 버전 | 지도·FDS 구성 | 주요 목적과 차이 |
| --- | --- | --- |
| `factory_v1` | 수작업으로 구성한 초기 공장 FDS 지도 | 열화상 카메라, MQ-135 CO 센서, 전체/부분 Costmap과 A*의 기본 동작을 검증한 기준 버전이다. |
| `factory_v2` | ROS SLAM 점유지도 `fire_demo_20260729_182054.pgm`을 0.2 m FDS 격자로 변환 | PGM의 occupied·unknown·no-go·기계 영역을 FDS 장애물로 자동 변환한다. 출구와 화재 시나리오를 `config/`와 `generated/`로 분리하고 지도 변환·검증 보고서를 제공한다. |
| `factory_v3` | `factory_v2` 구조를 회전해 EXIT1 축과 FDS X축을 정렬 | 센서 기반 Estimated Fire Map, 부분 Costmap, 안전 출구 평가, 요구조자 탐색·접근·추종, 경로 단순화, 동적 장애물, 화재 위치 추정, 이벤트 기반 재계획과 Pygame 통합 화면이 구현된 주 개발 버전이다. |
| `factory_v4` | `factory_v3` 장애물 지도에 로봇 크기의 FDS `OBST`를 추가한 `T_END=0` 구성 | 실제 로봇 형상과 시작 위치를 Smokeview/FDS에서 확인하기 위한 형상 검증 스냅샷이다. 전체 센서·임무 실행 버전을 대체하지 않는다. |
| `factory_v5` | 현재 ROS navigation 지도 `fire_robot_rpi/maps/inno_map_nav.pgm`을 기준으로 장애물 구조 재생성 | v3의 출구, 화재 확산, 센서, Costmap, MissionManager 및 대피 알고리즘을 유지하면서 SLAM과 FDS의 장애물 구조 불일치를 교정한 최종 버전이다. 관측 정보 노후화에 따른 제한적 불확실성 Cost도 포함한다. |

각 버전은 별도 폴더로 유지한다. 이전 버전을 수정해 새 버전을 흉내 내지 않으며,
최종 실행과 신규 개발은 특별한 이유가 없으면 `factory_v5`를 기준으로 한다.

## 2. factory_v5 알고리즘 동작 구성

### 전체 데이터 흐름

```text
FDS 온도·CO 결과
→ 검증된 NPZ 시계열 로딩
→ 로봇 pose에서 열화상 ray와 CO 센서값 생성
→ 센서로 실제 관측한 셀만 Estimated Fire Map에 기록
→ 정적 SLAM 지도 + 동적 장애물 + 온도·CO + 추정 화재 위험 결합
→ belief.final_cost_map 생성
→ 현재 임무 상태와 출구 상태 평가
→ A* 셀 경로 생성
→ 코너 waypoint 추출 및 안전 shortcut 검증
→ 검증된 world waypoint만 path follower에 전달
→ 이동 중 센서 갱신과 이벤트 기반 경로 재검증
```

Ground Truth 전체 온도·CO 지도는 가상 센서값 생성과 결과 평가에만 사용한다.
출구 선택과 경로 계획에는 로봇이 센서로 구성한 belief map만 사용한다.

### Costmap 구성

```text
final_cost_map
= 기본 이동 Cost
+ 관측 온도 Cost
+ 관측 CO Cost
+ 미관측 정보 Cost
+ 추정 화재 Cost
+ 관측 정보 노후화 Cost
```

- 온도 `60 °C` 이상 또는 CO `1600 ppm` 이상은 통행 불가(`inf`)다.
- 정적 장애물과 inflation 영역, 확인된 동적 장애물도 통행 불가다.
- 화재 위치 추정은 열화상 ray, 이동 중 CO 변화, 반복 관측을 융합하며 낮은
  확률만으로 셀을 차단하지 않는다.
- 관측 정보 노후화 Cost는 관측 후 5초까지 0이며, 이후 초당 0.05씩 증가해
  최대 2.0에서 멈춘다. 재관측하면 0으로 초기화되고, 노후화만으로 셀을
  차단하지 않는다.

### 임무와 출구 선택

1. 로봇은 설정된 초기 pose에서 0.8 m 전진한 뒤 탐색 경로를 만든다.
2. 현재 위치에서 아직 확인하지 않은 출구마다 A* 경로를 계산한다.
3. 안전하고 도달 가능한 경로 중 누적 위험 Cost와 경로 길이를 고려해 출구를
   선택한다.
4. 출구에 도착하면 접근 폭, 장애물, 온도·CO 위험을 검사해 `USABLE`,
   `BLOCKED`, `UNSAFE_FIRE` 등의 상태를 기록한다.
5. 모든 출구를 한 번에 고정된 순서로 방문하지 않고, 매 확인 이후 현재 위치와
   최신 Costmap에서 다음 출구를 다시 선택한다.
6. 진행 중 요구조자를 발견하면 출구 탐색을 중단하고 `APPROACH_VICTIM`으로
   전환한다.

### 요구조자 동행 대피

- 요구조자는 로봇 위치로 순간이동하지 않는다.
- 로봇이 실제로 지나온 pose history를 요구조자가 설정된 보행속도로 따라간다.
- 거리가 너무 멀어지면 `FOLLOW_WAIT`로 로봇을 정지시키고, 가까워진 뒤 남은
  경로를 다시 검증한 후 이동한다.
- 로봇과 요구조자가 모두 현재 `USABLE` 출구 반경에 도착해야 대피 성공이다.
- 이동 불가능한 요구조자는 동행시키지 않고 기존 구조대 알림 흐름으로 보낸다.

### 이벤트 기반 재계획

매 렌더링 프레임마다 A*를 실행하지 않는다. 다음과 같이 경로에 의미 있는
변화가 생겼을 때 남은 경로를 먼저 검증하고 필요한 경우에만 재계획한다.

- 현재 경로에 새 동적 장애물이 발생
- 경로 셀이 blocked 또는 `inf`로 변경
- 경로의 관측 온도·CO가 차단 임계값을 초과
- 현재 목적 출구가 `BLOCKED` 또는 `UNSAFE_FIRE`로 변경
- 더 안전한 출구가 최소 개선 조건과 cooldown을 만족
- 설정된 시간 또는 이동거리 기반 일반 재평가
- 요구조자 추종 실패 또는 통로 폭 부족

재계획 시 기존 이동을 먼저 정지하고, 최신 Costmap에서 새 경로를 생성·단순화·
검증한 다음에만 이동을 재개한다.

### 실행

```bash
cd ~/Robot_project/fire_robot/simulator/factory_v5
python3 run_partial_costmap_evacuation.py --no-thermal-window
```

GUI 없이 실행하려면 다음을 사용한다.

```bash
python3 run_partial_costmap_evacuation.py --headless
```

FDS 결과를 runtime NPZ로 다시 만들 때는 다음을 실행한다.

```bash
./prepare_temperature_npz.sh
python3 pack_co_to_npz.py
```

## 3. factory_v5 파일과 폴더 설명

### 주요 파일

| 경로 | 설명 |
| --- | --- |
| `factory_v5.fds` | FDS mesh, 반응, 출력 slice와 생성 geometry include를 연결하는 원본 입력 파일 |
| `run_partial_costmap_evacuation.py` | 센서, belief Costmap, MissionManager, 경로 계획, 이동과 GUI를 연결하는 최종 실행 진입점 |
| `run_evacuation_2d.py` | 열화상 3-D ray 대신 얇은 2-D 온도 slice 관측을 시험하는 대체 실행 진입점 |
| `validate_evacuation_setup.py` | FDS 범위, 지도 해상도, 시작점·요구조자·출구 좌표와 데이터 파일을 검사하는 도구 |
| `prepare_temperature_npz.sh` | FDS TEMP_3D 결과를 runtime 온도 NPZ로 변환하는 실행 스크립트 |
| `pack_temp3d_to_npz.py` | 온도 시계열을 `[time,z,y,x]` float32 NPZ로 저장하고 재검증하는 변환기 |
| `pack_co_to_npz.py` | CO slice를 ppm 단위 `[time,y,x]` NPZ로 변환하는 도구 |
| `record_simulation_replay.py` | 실제 계산 결과를 저장해 이후 일정한 속도로 재생할 수 있게 하는 recorder |
| `human_detection_sim.py` | 거리, 전방 FOV와 시야 가림을 적용한 요구조자 감지 모델 |
| `fds_temperature_io.py` | FDS 온도 CSV/좌표축 파싱과 검증 유틸리티 |
| `README.md` | v5 지도 생성, FDS 실행, NPZ 변환과 좌표계에 대한 상세 문서 |

### 폴더

| 폴더 | 설명 |
| --- | --- |
| `config/` | 로봇 시작 pose, 출구, 요구조자, 센서·Costmap 가중치, 재계획 및 동행 정책을 담은 YAML과 FDS 보조 설정 |
| `generated/` | 현재 `inno_map_nav.pgm`에서 변환한 장애물, 출구와 화재 시나리오 FDS include 파일 |
| `mapping/` | 정적/동적 점유지도, Estimated Fire Map, 부분 Costmap, 화재 위치 추정과 위험비용 계산 |
| `planner/` | weighted A*, 출구별 안전 평가와 최종 대피계획 생성 |
| `navigation/` | 경로 단순화, 이동 이력 복귀, 출구 전환, 탐색, 이벤트 기반 재계획과 요구조자 추종 정책 |
| `mission/` | 요구조자 탐색·접근·동행·대피 및 실패 상태를 관리하는 MissionManager |
| `world/` | 로봇, 요구조자, 출구, 장애물 entity와 전체 WorldState 및 명시적 결과 저장 구조 |
| `robot/` | 검증된 world waypoint를 따라가는 로봇 pose/path follower |
| `sensors/` | MLX90640 열화상, MQ-135 CO, mmWave 요구조자 감지 시뮬레이션 |
| `simulation/` | FDS/NPZ Ground Truth를 가상 센서에 제공하는 facade와 2-D thermal slice 모델 |
| `visualization/` | belief Costmap, 경로, 출구, 요구조자와 추정 화재를 표시하는 Pygame viewer |
| `examples/` | Pygame/FDS 전체 실행 없이 개별 전략과 자료구조를 확인하는 예제 |
| `tests/` | Costmap, A*, 출구 평가, 재계획, 탐색, 동행, 화재 추정과 통합 흐름의 단위·회귀 테스트 |
| `processed/` | 실행에 사용하는 온도·CO NPZ 시계열 |
| `csv_temp3d/` | fds2ascii 방식 온도 변환에서 사용하는 임시 CSV 디렉터리 |
| `output/` | 실제 이동 경로, replay 데이터와 시각화 결과 저장 위치 |
| `scripts/` | 현재 SLAM 지도에서 v5 장애물 geometry를 재생성하는 도구 |
| `validation/` | 지도 변환 hash, 회전 파라미터, geometry 검증 보고서와 overlay 이미지 |

FDS가 생성하는 `.smv`, `.sf`, `.s3d` 등의 대용량 원본 결과는 runtime 알고리즘
소스와 구분한다. 일반 시뮬레이션 실행에는 `processed/`의 검증된 NPZ가 있으면
되며, FDS 원본 결과는 NPZ를 다시 만들 때만 필요하다.
