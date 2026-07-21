# pack_temp3d_to_npz.py

import os
import re
import glob
import numpy as np

from test_fds_thermal_3d import load_fds2ascii_temperature_3d_csv


CSV_DIR = "csv_temp3d"
OUT_PATH = "processed/fds_temperature_3d_timeseries.npz"

# CSV를 npz로 묶은 뒤 원본 CSV를 지우고 싶으면 True
DELETE_CSV_AFTER_PACKING = True


def extract_time_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"_t(\d+)\.csv$", name)
    if m is None:
        raise ValueError(f"시간 정보를 파일명에서 찾지 못했습니다: {name}")
    return int(m.group(1))


def main():
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "factory_v1_temp_3d_t*.csv")))

    if len(csv_files) == 0:
        raise FileNotFoundError("csv_temp3d 폴더에서 CSV 파일을 찾지 못했습니다.")

    frames = []
    times = []

    xy_resolution = None
    z_resolution = None

    for path in csv_files:
        print("Loading:", path)

        t = extract_time_from_filename(path)

        temp_volume, xy_res, z_res = load_fds2ascii_temperature_3d_csv(path)

        frames.append(temp_volume.astype(np.float32))
        times.append(t)

        xy_resolution = xy_res
        z_resolution = z_res

    temperature = np.stack(frames, axis=0)
    times = np.array(times, dtype=np.float32)

    print()
    print("Final temperature array shape:")
    print("temperature.shape =", temperature.shape)
    print("times.shape =", times.shape)
    print("xy_resolution =", xy_resolution)
    print("z_resolution =", z_resolution)

    os.makedirs("processed", exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        temperature=temperature,
        times=times,
        xy_resolution=np.array(xy_resolution, dtype=np.float32),
        z_resolution=np.array(z_resolution, dtype=np.float32),
    )

    print()
    print("Saved:", OUT_PATH)

    if DELETE_CSV_AFTER_PACKING:
        for path in csv_files:
            os.remove(path)
        print("Original CSV files deleted.")


if __name__ == "__main__":
    main()
