# Offline tennis identification

Run on Ubuntu from `/home/ubuntu/mpc_sim`. The independent `.venv-tennis` keeps the existing MuJoCo `.venv` unchanged.

```bash
source project_env.sh
/usr/bin/python3 -m venv .venv-tennis
.venv-tennis/bin/python -m pip install -r tennis_calibration/requirements.lock.txt
bash run_tennis_calibration.sh
.venv-tennis/bin/python -m unittest discover -s tennis_calibration/tests -v
```

The raw archive is `data/tennis/raw/all_trajectories.zip`; it is never overwritten. New runs create a new `artifacts/tennis_calibration/<UTC timestamp>/` directory. Stage output schemas and assumptions are explicit in JSON. Trajectory IDs are complete ZIP member names, and inferred acquisition groups are held out whole.

`clean.py` records SI observations, keep/reason flags and provisional event labels. `flight.py` fits shared training drag and separates full-segment reconstruction from raw early-window forecasts. `bounce.py` fits effective restitution/tangential-speed ratios from independent guarded neighborhoods and evaluates the full causal forecast on fixed observed times. `plots.py` produces complete annotation review pages and diagnostic figures.

All coordinates, measurement reference, ground, spin and surface assumptions remain conditional. Effective tangential retention is not a measured friction coefficient, and measured contact height is not a calibrated ground plane. This package does not modify or deploy the MuJoCo robot model.

See [the Chinese calibration report](../docs/tennis_calibration_report.md) for measured parameters, held-out errors and limits.
