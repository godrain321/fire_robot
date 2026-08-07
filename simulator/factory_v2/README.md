# factory_v2: SLAM 기반 FDS 공장 지도

이 디렉터리는 `factory_v1`을 변경하지 않고, 지정된 ROS SLAM 점유격자를
FDS 고정 구조로 변환한다. 기존 발화점에서 가장 가까운 산업기계를 자동 선택하고,
그 기계에 인접한 누유 화재가 활성화되어 있다. 화원 설정, 요구조자와 동적 장애물은
`config/scenario.yaml`에 분리한다.

## 입력과 출력의 분리

- 원본(읽기 전용):
  `../../fire_robot_rpi/inno_jazzy_ws/maps/fire_demo_20260729_182054.{pgm,yaml}` 및
  `_semantic_points.yaml`, `../../fire_robot_rpi/maps/no_go_zones.yaml`
- 사람이 관리하는 설정: `config/`
- 자동 생성: `generated/`
- 검증 자료: `validation/`
- 향후 FDS 결과: `output/`
- 향후 센서용 후처리 결과: `processed/`

`generated/obstacles.inc`, `generated/exits.inc`, `generated/scenario.inc`,
`generated/occupancy_grid.npz`, 변환 보고서와 검증
이미지는 스크립트를 다시 실행하면 덮어쓴다. 원본 PGM/YAML은 수정하지 않는다.

## 좌표계

원본 YAML은 543×453셀, 0.05 m/셀, ROS 원점
`(-7.521, -17.712, 0)`이다. 회전 없는 지도이므로 FDS 원점을 SLAM 지도의
왼쪽 아래로 두었다.

```text
x_fds = x_ros - origin_x
y_fds = y_ros - origin_y

x_ros = origin_x + (col + 0.5) * resolution
y_ros = origin_y + (height - row - 0.5) * resolution
```

PGM은 위에서 아래로 `row`가 증가하므로 두 번째 식에서 Y를 반전한다. 셀
중심에는 `0.5 * resolution` 오프셋이 있으며, FDS 장애물 좌표는 coarse 셀의
경계 좌표다. 모든 변환은 `scripts/map_coordinates.py`에만 구현되어 있다.

Semantic point의 원본 ROS 값과 변환된 FDS 값은
`config/semantic_points.yaml`에 함께 기록했다. 변환 검증 시 이 파일을 원본
semantic YAML과 다시 비교한다.

## 점유 분류와 병합

ROS `trinary`, `negate: 0` 규칙에 따라 occupancy를 `(255-pixel)/255`로
계산한다.

- `occupancy > 0.65`: occupied
- `occupancy < 0.196`: free
- 그 사이: unknown

unknown은 외부 또는 사용하지 않는 공간으로 간주해 기본 차단한다. 먼저 0.10 m
허용오차의 contour approximation으로 직선에 가까운 SLAM 경계의 작은 픽셀
요철을 제거한다. 그 뒤 0.2 m coarse 셀이 occupied/unknown 셀 하나라도
포함하면 차단하는 `any_blocked` 정책을 적용한다. 최종 FDS geometry가 `OBST`
격자 기반이므로 대각선 벽에는 최소 한 셀 크기의 계단이 남는다.
원본 SLAM 파일은 수정하지 않는다. 생성 geometry에서만 0.10 m contour
허용오차로 순간적인 돌출과 짧은 요철을 제거할 수 있다. coarse 기준으로 열 수
있는 보정 셀은 최대 100개로 제한하며 실제 변경 수는 검증 보고서의
`inaccessible_area_preservation`에 기록한다. no-go polygon에는 이 보정을
적용하지 않고, EXIT 통로도 별도로 구분해 집계한다.

EXIT1~3은 외부 통로를 절개하거나 `OPEN VENT`를 만들지 않는다. 0.2 m coarse
셀을 잇는 높이 3.0 m의 파란 `EXIT_SOLID OBST`로 위치만 표시한다. EXIT1과
EXIT3은 각각 semantic 내부 중심을 통과하는 기울기 1.7647 선을 양쪽 SLAM
occupied 벽까지 연장한다. EXIT2는 FDS `(22.2, 12.0)`을 중심으로 기울기
-0.5667, 요청 길이 1.2 m인 선분이다. 격자화 결과는 계단형이며 설정은
`config/map_metadata.yaml`의 `exit_markers`에서 조정한다.

차단 격자의 각 행에서 연속 run을 찾고, 다음 행의 동일한 run과 합쳐
결정적인 직사각형 `OBST`를 만든다. 따라서 픽셀별 `OBST`가 아니다. occupied와
unknown은 각각 `SLAM_WALL`, `UNMAPPED_SOLID` 표면으로 구분되어 Smokeview에서
색으로 확인할 수 있다.

`fire_robot_rpi/maps/no_go_zones.yaml`의 ROS-map polygon은 같은 0.05 m 원본
격자에 rasterize한 뒤 `NO_GO_WALL`로 변환한다. no-go는 출구 절개보다 우선하며
어떤 셀도 자유 공간으로 바꾸지 않는다. 출구 생성 과정에서도 no-go나 기존
벽을 절개하지 않는다.

산업기계 중심은 원본 지도에서 모두 독립된 unknown 섬 내부에 있었다. 각 중심이
포함된 SLAM unknown 구성요소와 바로 인접한 1픽셀 occupied 윤곽을 기계 형상으로
사용한다. 직사각형 bounding box로 묶지 않으므로 원본 SLAM 윤곽이 보존된다.
인접 윤곽 범위는 `machinery_outline_dilation_cells`에서 조절한다. SLAM은 높이를
제공하지 않으므로 기계 높이는 `config/map_metadata.yaml`의
`machinery_height_m: 1.2`를 사용하는 임시값이다. 일반 벽과 no-go 벽은
3.0 m지만 기계는 1.2 m까지만 생성된다.

## FDS 영역과 계산량

- 원본 평면 범위: 27.15×22.65 m
- FDS mesh: 27.2×22.8×3.0 m
- 해상도: 0.2 m
- `IJK=136,114,15`
- 총 232,560셀

`factory_v1`의 0.5×0.5×0.25 m, 48,000셀보다 약 4.8배 크다. 0.1 m
등방 격자는 약 186만 셀로 v1의 약 39배이므로 초기 geometry 검증에는 쓰지
않았다. 원본의 27.15×22.65 m 범위를 넘는 +X/+Y padding은 차단한다.

## 재생성과 검증

저장소 루트에서:

```bash
cd simulator/factory_v2
python3 scripts/convert_slam_map.py
python3 scripts/validate_factory_map.py
python3 -m py_compile scripts/*.py
```

필요 패키지는 NumPy, Pillow, PyYAML, OpenCV다. 변환 스크립트를 반복 실행해도 정렬,
숫자 형식과 SHA-256이 동일하다. 검증 실패 시 두 번째 명령은 0이 아닌 종료
코드를 반환한다.

검증 자료:

- `generated/conversion_report.json`: 원본 해시, 셀 수, mesh와 OBST 수
- `validation/validation_report.json`: 좌표, 자유 공간, 경로, 여유 거리
- `validation/overlay.png`: 원본과 coarse 장애물 및 세 경로의 중첩
- `validation/fds_setup_test.txt`: 실제 FDS setup-only 결과

오버레이에서 어두운 적색은 occupied coarse cell, 청색은 unknown coarse cell,
주황색은 `NO_GO_WALL`, 진회색은 `MACHINERY`, 파란 막대는 고체 EXIT,
노랑은 INIT,
파랑/초록/자홍은 각 EXIT 경로다.
이 이미지는 지도 회전·반전과
벽 두께 변화를 확인하기 위한 자료이며 FDS/Smokeview 렌더링을 대신하지 않는다.

## FDS 실행

현재 `factory_v2.fds`의 `T_END`는 1100초다. 가장 늦게 점화되는 종이박스
셀도 약 480초의 연소시간을 확보하도록 확산시간과 연소시간을 함께 포함한다.
기계 인접 발화점은 5초에
설정 HRRPUA에 도달하고, 오일 표면을 따라 0.03 m/s로 번진다. 누유 바닥은
no-go 벽과 가까운 자유 셀을 우선하는 경로로 EXIT2까지 이어진다. 벽이나
no-go 장애물 표면을 태우지 않고 그 주변 바닥의 `VENT`만 연소시킨다.
추가 화재는 `NO_GO_ZONE_0050~0058`과 `SLAM_OCCUPIED_0211`, `0214`,
`0217`, `0219`, `0222`, `0225`, `0228`로 지정한 벽면의 인접 자유 바닥을
따라 `INDUSTRIAL_MACHINERY_04`까지 이어진다. 이 표면은 300 kW/m²,
0.008 m/s로 설정되어 오일보다
열방출률과 확산속도가 낮다. 갈색 검증 영역은 바닥의 가연성 박스를 의미하며
벽이나 기계 geometry를 변경하지 않는다.
`CATF`가 상대 경로를 사용하므로
`factory_v2` 디렉터리에서 실행한다.

```bash
cd simulator/factory_v2
ulimit -s unlimited
I_MPI_FABRICS=shm OMP_STACKSIZE=1G OMP_NUM_THREADS=4 \
  /home/park/FDS/FDS6/bin/fds_openmp factory_v2.fds
```

`ulimit -s unlimited`와 `OMP_STACKSIZE`는 큰 mesh의 화학종 밀도 계산에서 발생한
stack 부족 SIGSEGV를 방지한다. 같은 터미널에서 설정해야 한다.

FDS의 `CATF` 동작 때문에 결과 CHID는 `factory_v2_cat`이 된다. 소스와 결과를
엄격히 분리하려면 아래처럼 실행용 사본을 `output/setup`에 만든다.

```bash
mkdir -p output/setup/generated
cp factory_v2.fds output/setup/
cp generated/obstacles.inc output/setup/generated/
cp generated/exits.inc output/setup/generated/
cp generated/scenario.inc output/setup/generated/
cd output/setup
sed -i 's/T_END=1100/T_END=0.0/' factory_v2.fds
I_MPI_FABRICS=shm /home/park/FDS/FDS6/bin/fds factory_v2.fds
```

예상 핵심 출력은 `factory_v2_cat.out`, `factory_v2_cat.smv`와 slice 파일이다.
위 사본은 `T_END=0`이므로 geometry/setup 파일만 생성된다. 원본 입력은
`T_END=1100`으로 화재 계산을 수행한다. 화원 좌표와 성장 조건은 `factory_v2.fds`에 직접 넣지 않고
`scenario.yaml`에서 변경한 뒤 변환 스크립트로 `generated/scenario.inc`를 만든다.

## Smokeview 확인

FDS setup 후 결과 디렉터리에서 다음을 실행한다.

```bash
/home/park/FDS/FDS6/smvbin/smokeview factory_v2_cat
```

`Show/Hide > Geometry > Obstacles > Outline only`로 병합된 벽 윤곽을 확인하고,
전체 지도 방향, 직선화된 벽, 세 출구의 외부 통로와 기계 섬을 오버레이와
비교한다. slice가 생성된 실제 시나리오에서는 우클릭 `Load/Unload` 메뉴에서
온도/CO slice를 불러온다.

현재 환경에는 Smokeview 6.11.0이 설치되어 있고 `.smv` 생성까지 확인했지만,
그래픽 display/Xvfb가 없어 사람이 보는 최종 화면 검사는 수행하지 못했다.

기본 stack 제한에서는 첫 timestep의 `mass_mp_density_`에서 SIGSEGV가 발생했다.
stack 제한 해제 후 최종 geometry와 화학종을 포함한 setup-only 검사가
정상 완료됐다. 산업기계 인접 발화용 오일 VENT 22개와 지정 벽면을 따르는
열반응형 종이박스 적재물 OBST 23개가 생성된다. 천장 ZMAX 경계에는 2 m × 2 m
`OPEN` 환기구가 생성된다. 2초 단축 실행에서 입력 파싱, 초기화 및 계산 완료를
확인했지만 전체 1100초 화재 계산은 아직 완주 검증하지 않았다.

## 공식 매뉴얼과 factory_v1 재사용 근거

`docs/FDS_User_Guide.pdf`에서 다음 항목을 확인했다.

- 5.4 `CATF`: 여러 입력 파일 삽입, 최대 20개 파일, 줄 길이 400자 제한,
  결과 CHID의 `_cat` suffix
- 6.2 `TIME`: `T_END=0` setup-only geometry 검사
- 6.3 `MESH`: `XB`, `IJK`, 오른손 좌표계와 가능한 등방 셀 권장
- 7.2 `OBST`: 직육면체 장애물과 mesh 경계 clipping
- 7.4 `VENT`: 바닥 누유 화재 표면 및 외부 mesh 경계의 `OPEN` 환기구
- 9.2.3 `MATL`, `SURF`, `IGNITION_TEMPERATURE`: 고체의 열적 물성과
  표면온도 기반 착화 후 지정 연소율 적용
- 9.3.8 `BURN_DURATION`: 각 표면 셀이 착화된 이후의 연소 유지시간
- 22.1/22.4 `DUMP`, `SLCF`: slice 주기, 3-D slice와 `CELL_CENTERED`

`docs/SMV_User_Guide.pdf` 1.3.2의 command-line 실행과 `Load/Unload`, 그리고
13장의 FDS 입력 geometry 디버깅 절차를 참고했다. Verification/Technical
Reference 문서는 파일 존재와 버전을 확인했지만 이번 입력 문법의 근거로는
User Guide를 우선했다.

`factory_v1`에서는 높이 5 m, `WALL` 계열 표면, propane `REAC`, 전체 3-D 온도
`SLCF`, z=1.3 m CO `SLCF`, `DUMP` 구조를 재사용했다. 기존 화원 위치, rack,
office, 단일 출구 좌표는 새 SLAM 지도와 맞지 않아 재사용하지 않았다. v1의
propane `REAC`와 성장곡선 형식을 산업기계 누유 화원에 재사용했다.

## 현재 제한과 조정할 값

- `wall_height_m: 3.0`: SLAM에 높이가 없어 임시값
- `machinery_height_m: 1.2`: 산업기계 높이 측정값이 없어 분리한 임시값
- `machinery_outline_dilation_cells: 1`: SLAM 기계 본체에 포함할 인접 윤곽 범위
- `fds_height_m: 3.0`: 벽·기둥 높이와 동일한 천장 높이
- `fds_resolution_m: 0.20`: 계산량과 벽 보존 사이의 선택
- unknown을 전부 차단하므로 실제로 사용 가능한 미탐색 공간도 막힐 수 있음
- no-go polygon 수정 시 변환과 연결성 검증을 다시 실행해야 함
- 산업기계 형상은 SLAM unknown/occupied 윤곽이며 실제 치수 측정값이 아님
- 고체 출구 폭 1.2 m, 두께 0.2 m, 높이 3.0 m는 semantic 원본에 치수가
  없어 조정 가능한 임시값
- contour 단순화 허용오차 0.10 m도 실제 벽 측량 결과에 따라 조정 필요
- 화재 HRRPUA 1000 kW/m², 누유 반경 1.2 m와 확산속도 0.03 m/s는 조정 가능한 값
- 벽면 종이박스 경로의 no-go 구간 반폭은 0.20 m, SLAM wall 구간 반폭은
  0.40 m다. 예전의 시간 기반 `SPREAD_RATE`는 사용하지 않는다.
- 적층 종이박스는 유효 밀도 150 kg/m³, 비열 1.40 kJ/(kg·K), 열전도율
  0.08 W/(m·K), 표면 열두께 6 mm, 높이 0.60 m의 조정 가능한 근사 모델이다.
  표면이 250 °C에 도달한 셀만 착화하며 20초 동안 300 kW/m²까지 증가한 뒤
  최소 480초 동안 연소한다. 골판지 종류·수분·적층 공극에 따라 물성이 크게
  달라지므로 실물 시험값이 생기면 `config/scenario.yaml`에서 보정해야 한다.
- 천장 환기구는 `[2,4]×[18,20] m`, z=3 m이며 `roof_vent.xb`로 변경할 수 있다.
  `OPEN`은 반드시 계산영역 외부 경계에 있어야 한다.
- SLAM wall에서 산업기계 2로 내려오는 연결 화재는 요청 기울기 1.70이며,
  0.2 m 격자화 결과 실제 기울기는 약 1.714다.
- 화재만 재생성할 때는 `python3 scripts/generate_fire_scenario.py`를 사용한다.
  보호 해시를 먼저 확인하며 `scenario.inc`와 화재 보고서만 수정한다.
  `obstacles.inc`와 `exits.inc`는 읽기 전용이다.
- no-go 벽 추종 가중치, 화재 띠 반폭 0.40 m, EXIT2 차단 반경 0.60 m도
  `config/scenario.yaml`에서 조정 가능
- 요구조자와 동적 장애물은 아직 미정
- Smokeview 육안 검증은 display가 있는 환경에서 수행 필요
