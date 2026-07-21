import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import fdsreader as fds


# ============================================================
# MQ-135 SENSOR MODEL
# ============================================================

@dataclass
class MQ135Config:
    min_ppm: float = 10.0
    max_ppm: float = 1000.0

    warning_ppm: float = 700.0
    danger_ppm: float = 900.0
    blocked_ppm: float = 950.0

    response_tau: float = 2.0

    noise_std_ratio: float = 0.03
    noise_std_min: float = 5.0

    vc: float = 5.0
    rl: float = 10_000.0
    rs_100ppm: float = 10_000.0
    alpha: float = 0.6


@dataclass
class MQ135Reading:
    true_ppm: float
    measured_ppm: float
    voltage: float
    cost: float
    saturated: bool
    status: str


class MQ135Sensor:
    def __init__(self, config=None):
        self.config = config if config is not None else MQ135Config()
        self.filtered_ppm = self.config.min_ppm

    def read(self, true_ppm: float, dt: float) -> MQ135Reading:
        cfg = self.config

        true_ppm = max(0.0, float(true_ppm))

        saturated = true_ppm >= cfg.max_ppm
        target_ppm = np.clip(true_ppm, cfg.min_ppm, cfg.max_ppm)

        if dt <= 0:
            response_alpha = 1.0
        else:
            response_alpha = 1.0 - math.exp(-dt / cfg.response_tau)

        self.filtered_ppm += response_alpha * (target_ppm - self.filtered_ppm)

        noise_std = max(cfg.noise_std_min, abs(self.filtered_ppm) * cfg.noise_std_ratio)
        noisy_ppm = self.filtered_ppm + np.random.normal(0.0, noise_std)

        measured_ppm = float(np.clip(noisy_ppm, cfg.min_ppm, cfg.max_ppm))
        voltage = self._ppm_to_voltage(measured_ppm)
        cost = self.ppm_to_cost(measured_ppm)
        status = self.ppm_to_status(measured_ppm)

        return MQ135Reading(
            true_ppm=true_ppm,
            measured_ppm=measured_ppm,
            voltage=voltage,
            cost=cost,
            saturated=saturated,
            status=status,
        )

    def ppm_to_cost(self, ppm: float) -> float:
        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return float("inf")

        normalized = ppm / cfg.blocked_ppm
        return float(1.0 + 99.0 * (normalized ** 2))

    def ppm_to_status(self, ppm: float) -> str:
        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return "BLOCKED"
        elif ppm >= cfg.danger_ppm:
            return "DANGER"
        elif ppm >= cfg.warning_ppm:
            return "WARNING"
        elif ppm >= 300.0:
            return "CAUTION"
        else:
            return "SAFE"

    def _ppm_to_voltage(self, ppm: float) -> float:
        cfg = self.config

        ppm = max(cfg.min_ppm, min(ppm, cfg.max_ppm))
        rs = cfg.rs_100ppm * ((100.0 / ppm) ** cfg.alpha)
        voltage = cfg.vc * cfg.rl / (rs + cfg.rl)

        return float(voltage)


# ============================================================
# ROBOT PATH
# ============================================================

def interpolate_waypoints(waypoints, speed_mps, times):
    """
    주어진 waypoints를 따라 로봇 위치를 시간별로 계산한다.
    speed_mps: 로봇 속도 [m/s]
    times: FDS 출력 시간 배열
    """

    waypoints = np.asarray(waypoints, dtype=float)

    segments = []
    total_length = 0.0

    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        length = np.linalg.norm(p1 - p0)
        segments.append((p0, p1, length))
        total_length += length

    positions = []

    for t in times:
        dist = speed_mps * t

        if dist >= total_length:
            positions.append(waypoints[-1])
            continue

        acc = 0.0
        for p0, p1, length in segments:
            if acc + length >= dist:
                ratio = (dist - acc) / length
                pos = p0 + ratio * (p1 - p0)
                positions.append(pos)
                break
            acc += length

    return np.asarray(positions)


# ============================================================
# FDS SLICE READING
# ============================================================

def find_co_slice(sim):
    """
    FDS 결과에서 CO_Z130 또는 CARBON MONOXIDE / VOLUME FRACTION slice를 찾는다.
    """

    print("\n[INFO] FDS slices found:")
    for i, slc in enumerate(sim.slices):
        print(f"  [{i}] {slc}")

    # 1순위: ID 문자열에 CO_Z130이 들어간 slice
    for slc in sim.slices:
        text = str(slc).upper()
        if "CO_Z130" in text:
            print("\n[INFO] Selected slice by ID: CO_Z130")
            return slc

    # 2순위: CARBON MONOXIDE 또는 VOLUME FRACTION 포함 slice
    candidates = []
    for slc in sim.slices:
        text = str(slc).upper()
        if "CARBON MONOXIDE" in text or "VOLUME FRACTION" in text:
            candidates.append(slc)

    if len(candidates) == 1:
        print("\n[INFO] Selected CO slice by quantity.")
        return candidates[0]

    if len(candidates) > 1:
        print("\n[WARNING] Multiple CO-like slices found. Using the first one.")
        return candidates[0]

    raise RuntimeError(
        "CO slice를 찾지 못했어. .fds 파일에 아래 SLCF가 있는지 확인해:\n"
        "&SLCF ID='CO_Z130', PBZ=1.30, QUANTITY='VOLUME FRACTION', "
        "SPEC_ID='CARBON MONOXIDE', CELL_CENTERED=.TRUE. /"
    )


def load_global_slice(slc):
    """
    fdsreader의 to_global()을 사용해서 slice data와 좌표를 가져온다.
    """

    result = slc.to_global(
        masked=True,
        fill=np.nan,
        return_coordinates=True
    )

    # 보통 (data, coordinates) 형태로 반환됨
    if isinstance(result, tuple) and len(result) == 2:
        data, coords = result
    else:
        raise RuntimeError("slice.to_global(return_coordinates=True) 반환 형식을 확인해야 해.")

    data = np.asarray(data, dtype=float)

    # coords는 dict 형태인 경우가 많음
    x_coords = np.asarray(coords["x"], dtype=float)
    y_coords = np.asarray(coords["y"], dtype=float)

    return data, x_coords, y_coords


def sample_xy(data_2d, x_coords, y_coords, x, y):
    """
    x, y 위치에서 가장 가까운 cell의 값을 읽는다.
    data_2d는 x-y 평면 slice라고 가정한다.
    """

    ix = int(np.nanargmin(np.abs(x_coords - x)))
    iy = int(np.nanargmin(np.abs(y_coords - y)))

    # fdsreader slice는 보통 data[x_index, y_index] 구조
    if data_2d.shape[0] == len(x_coords) and data_2d.shape[1] == len(y_coords):
        return float(data_2d[ix, iy])

    # 혹시 data[y_index, x_index] 구조로 잡힌 경우를 대비
    if data_2d.shape[0] == len(y_coords) and data_2d.shape[1] == len(x_coords):
        return float(data_2d[iy, ix])

    raise RuntimeError(
        f"좌표 크기와 data shape이 맞지 않아. data shape={data_2d.shape}, "
        f"x={len(x_coords)}, y={len(y_coords)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fds-dir", type=str, default=".", help="FDS 결과 폴더")
    parser.add_argument("--speed", type=float, default=0.45, help="로봇 속도 [m/s]")
    parser.add_argument("--realtime", action="store_true", help="실시간처럼 천천히 출력")
    parser.add_argument("--delay", type=float, default=0.05, help="realtime 출력 delay [s]")
    args = parser.parse_args()

    fds_dir = Path(args.fds_dir).resolve()

    print(f"[INFO] Loading FDS result folder: {fds_dir}")

    sim = fds.Simulation(str(fds_dir))
    co_slice = find_co_slice(sim)

    co_data, x_coords, y_coords = load_global_slice(co_slice)

    times = np.asarray(co_slice.times, dtype=float)

    print("\n[INFO] CO slice loaded")
    print(f"  data shape: {co_data.shape}")
    print(f"  time steps: {len(times)}")
    print(f"  x range: {x_coords.min():.2f} ~ {x_coords.max():.2f} m")
    print(f"  y range: {y_coords.min():.2f} ~ {y_coords.max():.2f} m")

    # ------------------------------------------------------------
    # 로봇 이동 경로
    # 공장 왼쪽 아래 내부 지점에서 시작해서 화재원 중심으로 이동
    #
    # 화재원:
    # XB = 16.0~17.0, 25.0~26.0
    # 중심 = 16.5, 25.5
    #
    # 선반을 피하기 위해 왼쪽 통로를 따라 위로 올라간 뒤,
    # 랙 위쪽 공간을 통해 화재원으로 접근하는 간단한 waypoint 경로
    # ------------------------------------------------------------

    waypoints = [
        (1.0, 1.0),      # 왼쪽 아래 내부 시작점
        (1.0, 21.0),     # 왼쪽 통로를 따라 위로 이동
        (7.0, 21.0),     # 랙 위쪽 공간으로 이동
        (16.5, 25.5),    # 화재원 중심 근처
    ]

    positions = interpolate_waypoints(
        waypoints=waypoints,
        speed_mps=args.speed,
        times=times
    )

    sensor = MQ135Sensor()

    output_rows = []

    print("\n[START] MQ-135 gas sensor replay")
    print("time[s], x[m], y[m], true_CO[ppm], measured[ppm], voltage[V], status, cost")

    prev_time = times[0]

    for it, t in enumerate(times):
        dt = float(t - prev_time) if it > 0 else 0.0
        prev_time = t

        x, y = positions[it]

        # FDS VOLUME FRACTION은 mol/mol 형태라고 보고 ppm으로 변환
        co_volume_fraction = sample_xy(
            data_2d=co_data[it],
            x_coords=x_coords,
            y_coords=y_coords,
            x=x,
            y=y
        )

        if np.isnan(co_volume_fraction):
            true_ppm = 0.0
        else:
            true_ppm = co_volume_fraction * 1_000_000.0

        reading = sensor.read(true_ppm=true_ppm, dt=dt)

        cost_str = "inf" if math.isinf(reading.cost) else f"{reading.cost:.2f}"

        print(
            f"{t:6.1f}, "
            f"{x:5.2f}, {y:5.2f}, "
            f"{reading.true_ppm:9.2f}, "
            f"{reading.measured_ppm:9.2f}, "
            f"{reading.voltage:6.3f}, "
            f"{reading.status:8s}, "
            f"{cost_str}"
        )

        output_rows.append({
            "time_s": t,
            "x_m": x,
            "y_m": y,
            "true_co_ppm": reading.true_ppm,
            "measured_ppm": reading.measured_ppm,
            "voltage_v": reading.voltage,
            "status": reading.status,
            "cost": reading.cost,
            "saturated": reading.saturated,
        })

        if args.realtime:
            time.sleep(args.delay)

    # ------------------------------------------------------------
    # CSV 저장
    # ------------------------------------------------------------

    csv_path = fds_dir / "mq135_path_log.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "x_m",
                "y_m",
                "true_co_ppm",
                "measured_ppm",
                "voltage_v",
                "status",
                "cost",
                "saturated",
            ]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n[SAVED] CSV log: {csv_path}")

    # ------------------------------------------------------------
    # 그래프 저장
    # ------------------------------------------------------------

    t_list = [row["time_s"] for row in output_rows]
    true_list = [row["true_co_ppm"] for row in output_rows]
    measured_list = [row["measured_ppm"] for row in output_rows]

    plt.figure()
    plt.plot(t_list, true_list, label="True CO ppm from FDS")
    plt.plot(t_list, measured_list, label="MQ-135 measured ppm")
    plt.axhline(950, linestyle="--", label="Blocked threshold 950 ppm")
    plt.xlabel("Time [s]")
    plt.ylabel("CO concentration [ppm]")
    plt.title("MQ-135 Gas Sensor Replay Along Robot Path")
    plt.legend()
    plt.grid(True)

    plot_path = fds_dir / "mq135_path_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Plot: {plot_path}")


if __name__ == "__main__":
    main()