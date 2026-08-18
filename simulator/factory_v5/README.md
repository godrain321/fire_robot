# factory_v5

## Geometry source and scope

`factory_v5` preserves the `factory_v3` simulation, exit, blocker, and fire
scenario settings. Only the SLAM-derived obstacle geometry is replaced.

The obstacle source is the current ROS navigation map:

- `../../fire_robot_rpi/maps/inno_map_nav.pgm`
- size: `543 x 453` cells
- resolution: `0.05 m/cell`
- ROS origin: `[-7.521, -17.712, 0.0]`

It is downsampled to the existing `0.20 m` FDS grid and transformed with the
same rotation, pivot, translation, and domain used by `factory_v3`.
`generated/source_obstacles_unrotated.inc` is the intermediate map conversion;
`generated/obstacles.inc` is the final rotated geometry.

These files intentionally remain byte-identical to `factory_v3`:

- `generated/exits.inc`
- `generated/scenario.inc`
- `config/exit_blockers.inc`

## Portable runtime data

`run_partial_costmap_evacuation.py` uses the versioned sensor Ground Truth
archives below and does not require raw FDS `.smv`, `.sf`, or `.s3d` files:

- `processed/fds_temperature_3d_timeseries.npz`
- `processed/fds_co_2d_timeseries.npz`

Run the simulator from this directory with:

```bash
python3 run_partial_costmap_evacuation.py --no-thermal-window
```

Raw FDS results are only needed when regenerating these archives. Use
`prepare_temperature_npz.sh` for temperature and `pack_co_to_npz.py` for CO.
The temperature helper reads the complete cell-centred FDS slice directly and
therefore follows the actual result time range (currently 0..600 s) without
creating hundreds of temporary CSV files.

`factory_v5` is a rigidly rotated derivative of the current `factory_v2`. The
principal axis through the existing EXIT1 solid marker is rotated onto the
positive FDS X axis. Nothing below `factory_v2` is written by the generator.

The active v5 domain is the rectangle `x=1.8..30.0 m`, `y=5.6..28.0 m`.
`UNMAPPED_SOLID` is absent from the v5 output. Its rotated occupancy is used
only to classify the exterior and is emitted as the separate `OTHERS` surface;
the padding between the rotated source extent and rectangular domain is also
`OTHERS`. The building interior remains empty space.
A one-cell SLAM-wall bridge at `x=12.4..12.6`, `y=6.4..6.6 m` restores the
connection identified between the former v3 wall records 0087 and 0094.

## Regenerate

From the repository root:

```bash
python3 simulator/factory_v5/scripts/generate_factory_v5_obstacles.py
```

The script reads the committed current-map intermediate obstacle geometry and
inverse-samples it onto the rotated 0.2 m grid. It only rewrites
`generated/obstacles.inc`; hashes ensure exits, fire scenario, and exit blockers
are not changed.

## Run FDS

```bash
cd simulator/factory_v5
ulimit -s unlimited
I_MPI_FABRICS=shm OMP_STACKSIZE=1G OMP_NUM_THREADS=4 \
  /home/park/FDS/FDS6/bin/fds_openmp factory_v5.fds
```

Because `CATF` is used, the resulting CHID is `factory_v5_cat`.

## FDS result processing and evacuation simulation

The factory_v1 sensor/costmap/A* flow is copied locally under `mapping/`,
`planner/`, `robot/`, `sensors/`, `simulation/`, and `visualization/`. Runtime
code does not import factory_v1. Factory-v3-specific mission positions and paths
are in `config/evacuation.yaml`.

The current FDS input provides `TEMP_3D` every 1 s over the complete mesh and a
cell-centered `CO_Z130` volume-fraction slice at z=1.3 m. Run the long FDS case
only when ready, using one OpenMP thread first:

```bash
cd ~/Robot_project/fire_robot/simulator/factory_v5
ulimit -s unlimited
export OMP_STACKSIZE=2G
export OMP_NUM_THREADS=1
export I_MPI_FABRICS=shm
/home/park/FDS/FDS6/bin/fds_openmp factory_v5.fds
```

After FDS completes, create both runtime archives:

```bash
./prepare_temperature_npz.sh
python3 pack_co_to_npz.py
```

`export_temp3d_all.sh` remains available as a slower diagnostic fallback. It
reads `T_END` from `factory_v5.fds`, selects the result CHID
`factory_v5_cat`, and refuses to overwrite an existing CSV frame.

The packer sorts frame times numerically, requires a complete and identical
Cartesian coordinate grid in every CSV, stores `temperature[time,z,y,x]`, and
reopens a temporary NPZ to validate shape, dtype, sorted times, NaNs, minimum,
and maximum. Only after successful validation is the temporary file atomically
moved to `processed/fds_temperature_3d_timeseries.npz`; only the exact input
files used from `csv_temp3d/` are then deleted. Use `--keep-csv` to retain them.
The convenience `prepare_temperature_npz.sh` performs both commands with the
same safety checks.

Validate paths and coordinates without result data:

```bash
python3 validate_evacuation_setup.py
```

After both the temperature NPZ and final FDS CO slice exist, run:

```bash
python3 run_partial_costmap_evacuation.py
```

Use `--headless` to disable Pygame and Matplotlib. The runtime contracts are:

- NPZ temperature: `[time,z,y,x]`, Celsius
- thermal camera input and obstacle volume: `[z,y,x]`
- CO Ground Truth: `[time,y,x]`, converted from mol/mol to ppm by `×1e6`
- planner/costmap arrays: `[y,x]`; planner nodes: `(x,y)`
- Pygame: world Y increases upward and is explicitly flipped for the screen

The thermal camera retains 32×24 pixels, 110°×75° FOV, 0.30 m height and front
offset, 0.20..7.0 m range, and `measurement_mode="max"`. Temperature ≥60 °C
or measured CO ≥1600 ppm blocks a belief cell. Human detection retains the
factory_v1 10 m range and line-of-sight test. The robot approaches within 1 m,
then chooses the currently reachable configured exit with minimum weighted-A*
cost.

Observed fire-map cells also carry a bounded information-age uncertainty cost.
The default policy adds no cost for 5 seconds, then adds 0.05 per second up to
2.0. A fresh temperature or CO observation resets the cell's age cost. Aging
never changes the saved sensor value and never creates a blocked cell.

`robot_start`, the victim, and exit approach points are explicit temporary
values in `config/evacuation.yaml`. They are validated against the inflated
obstacle grid and are never silently moved. The semantic INIT and exit marker
centers themselves are blocked by current machinery/solid marker geometry, so
the temporary free approach points must be replaced when measured mission
positions are available.

## Coordinate conversion

`validation/rotation_report.json` records the measured EXIT1 angle, applied
rotation, pivot, translation, and final mesh bounds. Rotated INIT, EXIT, and
machinery coordinates are in `config/semantic_points.yaml` under `fds_v3`.
The original ROS and factory_v2 coordinates are retained for traceability.

The `XB` form of FDS `OBST` is axis-aligned and has no arbitrary rotation. For
that reason, directly rotating rectangle corners and using their bounding boxes
would thicken walls. This generator instead rotates the occupancy grid and then
merges cells. `validation/rotation_overlay.png` provides a visual check.
