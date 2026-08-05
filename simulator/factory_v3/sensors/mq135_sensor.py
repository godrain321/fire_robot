from dataclasses import dataclass
import math
import numpy as np


@dataclass
class MQ135Config:
    """
    MQ-135 가상 센서 설정값

    사진 속 사양 기준:
    - 검출 농도: 10 ~ 1000 ppm
    - 표준 테스트 조건: 20°C, 60%RH
    - 가열 전압: 5.0V 근처
    - 센서 저항 Rs: 100ppm NH3 기준 약 2kΩ ~ 20kΩ
    """

    # MQ-135 사양상 검출 가능 범위
    min_ppm: float = 10.0
    max_ppm: float = 1000.0

    # cost map에서 위험 지역으로 볼 기준
    warning_ppm: float = 700.0
    danger_ppm: float = 900.0
    blocked_ppm: float = 950.0

    # 센서 반응 지연 시간 상수
    # 값이 클수록 센서값이 천천히 변함
    response_tau: float = 2.0

    # 노이즈 설정
    noise_std_ratio: float = 0.03   # 측정값의 3% 정도 노이즈
    noise_std_min: float = 5.0      # 최소 ±5ppm 정도 노이즈

    # 표준 환경 조건
    standard_temp_c: float = 20.0
    standard_humidity: float = 60.0

    # 온습도 보정 계수
    # 실제 MQ 센서는 온습도 영향을 크게 받지만,
    # 여기서는 간단한 시뮬레이션용 보정만 적용
    temp_coeff: float = 0.005       # 1°C당 0.5% 변화
    humidity_coeff: float = 0.003   # 1%RH당 0.3% 변화

    # 전압 출력 시뮬레이션용 값
    vc: float = 5.0                 # 회로 전압
    rl: float = 10_000.0            # 부하 저항 10kΩ 가정
    rs_100ppm: float = 10_000.0     # 100ppm에서 센서 저항 10kΩ 가정
    alpha: float = 0.6              # 사양상 농도 기울기 alpha <= 0.6


@dataclass
class MQ135Reading:
    """
    MQ-135 센서가 한 번 측정했을 때의 결과
    """

    true_ppm: float          # 실제 환경의 가스 농도
    measured_ppm: float      # MQ-135가 읽은 농도
    voltage: float           # 아날로그 출력 전압 가정값
    cost: float              # cost map에 사용할 cost
    saturated: bool          # 센서가 1000ppm 근처에서 포화되었는지
    status: str              # SAFE / CAUTION / WARNING / DANGER / BLOCKED


class MQ135Sensor:
    def __init__(self, config: MQ135Config | None = None):
        self.config = config if config is not None else MQ135Config()

        # 센서 내부 필터값
        # 처음에는 깨끗한 공기 수준이라고 가정
        self.filtered_ppm = self.config.min_ppm

    def read(
        self,
        true_ppm: float,
        dt: float,
        temperature_c: float = 20.0,
        humidity: float = 60.0,
    ) -> MQ135Reading:
        """
        실제 가스 농도 true_ppm을 넣으면,
        MQ-135가 실제로 읽을 법한 measured_ppm을 반환한다.

        Parameters
        ----------
        true_ppm:
            시뮬레이션 상 실제 가스 농도.
            FDS나 가스맵에서 가져온 값.

        dt:
            시뮬레이션 시간 간격.
            예: 0.1초마다 업데이트하면 dt=0.1

        temperature_c:
            현재 온도.

        humidity:
            현재 상대습도.
        """

        cfg = self.config

        # 1. 음수 방지
        true_ppm = max(0.0, float(true_ppm))

        # 2. 온습도 영향 간단 보정
        temp_error = temperature_c - cfg.standard_temp_c
        humidity_error = humidity - cfg.standard_humidity

        env_factor = 1.0
        env_factor += cfg.temp_coeff * temp_error
        env_factor += cfg.humidity_coeff * humidity_error

        env_factor = max(0.5, min(env_factor, 1.5))

        compensated_ppm = true_ppm * env_factor

        # 3. MQ-135 측정 범위 적용
        # 실제 농도가 1000ppm을 넘어도 MQ-135는 그 이상을 정확히 구분하지 못한다고 가정
        saturated = compensated_ppm >= cfg.max_ppm

        target_ppm = np.clip(
            compensated_ppm,
            cfg.min_ppm,
            cfg.max_ppm
        )

        # 4. 센서 반응 지연 적용
        # 급격히 농도가 변해도 센서값은 천천히 따라가게 만듦
        if dt <= 0:
            response_alpha = 1.0
        else:
            response_alpha = 1.0 - math.exp(-dt / cfg.response_tau)

        self.filtered_ppm += response_alpha * (target_ppm - self.filtered_ppm)

        # 5. 노이즈 추가
        noise_std = max(
            cfg.noise_std_min,
            abs(self.filtered_ppm) * cfg.noise_std_ratio
        )

        noisy_ppm = self.filtered_ppm + np.random.normal(0.0, noise_std)

        # 6. 최종 측정값도 센서 범위 안으로 제한
        measured_ppm = float(np.clip(noisy_ppm, cfg.min_ppm, cfg.max_ppm))

        # 7. 아날로그 전압값 계산
        voltage = self._ppm_to_voltage(measured_ppm)

        # 8. cost 계산
        cost = self.ppm_to_cost(measured_ppm)

        # 9. 상태 문자열
        status = self.ppm_to_status(measured_ppm)

        return MQ135Reading(
            true_ppm=true_ppm,
            measured_ppm=measured_ppm,
            voltage=voltage,
            cost=cost,
            saturated=saturated,
            status=status
        )

    def read_from_map(
        self,
        gas_map: np.ndarray,
        robot_x: float,
        robot_y: float,
        resolution: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        dt: float = 0.1,
        temperature_c: float = 20.0,
        humidity: float = 60.0,
    ) -> MQ135Reading:
        """
        2D 가스 농도 map에서 로봇 위치의 ppm을 읽어서 센서값을 반환한다.

        gas_map[y, x] 형태라고 가정.
        단위는 ppm이라고 가정.
        """

        true_ppm = self._sample_map_nearest(
            gas_map=gas_map,
            robot_x=robot_x,
            robot_y=robot_y,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y
        )

        return self.read(
            true_ppm=true_ppm,
            dt=dt,
            temperature_c=temperature_c,
            humidity=humidity
        )

    def ppm_to_cost(self, ppm: float) -> float:
        """
        MQ-135 측정값을 cost map 값으로 변환.

        950ppm 이상이면 진입 금지로 처리.
        A* 코드에서 float('inf') 처리가 불편하면
        1e9 같은 큰 값으로 바꿔도 됨.
        """

        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return float("inf")

        # 0~blocked_ppm 사이를 부드럽게 증가시키는 방식
        # 농도가 높아질수록 cost가 빠르게 커짐
        normalized = ppm / cfg.blocked_ppm
        cost = 1.0 + 99.0 * (normalized ** 2)

        return float(cost)

    def ppm_to_status(self, ppm: float) -> str:
        """
        ppm 값을 사람이 보기 쉬운 상태로 변환.
        """

        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return "BLOCKED"
        elif ppm >= cfg.danger_ppm:
            return "DANGER"
        elif ppm >= cfg.warning_ppm:
            return "WARNING"
        elif ppm >= 300:
            return "CAUTION"
        else:
            return "SAFE"

    def _ppm_to_voltage(self, ppm: float) -> float:
        """
        MQ-135 아날로그 출력 전압을 매우 단순하게 흉내냄.

        실제 MQ 센서는 Rs/R0 그래프를 보고 보정해야 하지만,
        여기서는 시뮬레이션용으로 다음처럼 가정한다.

        - 100ppm에서 Rs = 10kΩ
        - 농도가 증가하면 Rs 감소
        - Vout = Vc * RL / (Rs + RL)
        """

        cfg = self.config

        ppm = max(cfg.min_ppm, min(ppm, cfg.max_ppm))

        # 농도 증가 시 Rs 감소
        rs = cfg.rs_100ppm * ((100.0 / ppm) ** cfg.alpha)

        # 전압 분배 공식
        voltage = cfg.vc * cfg.rl / (rs + cfg.rl)

        return float(voltage)

    def _sample_map_nearest(
        self,
        gas_map: np.ndarray,
        robot_x: float,
        robot_y: float,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> float:
        """
        로봇 위치에서 가장 가까운 grid cell의 가스 농도 읽기.
        """

        col = int((robot_x - origin_x) / resolution)
        row = int((robot_y - origin_y) / resolution)

        height, width = gas_map.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            # 맵 밖이면 위험하게 보거나 0으로 볼 수 있는데,
            # 여기서는 맵 밖을 진입 금지 쪽으로 보는 게 안전함.
            return self.config.max_ppm

        return float(gas_map[row, col])