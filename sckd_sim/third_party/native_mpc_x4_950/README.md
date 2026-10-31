# X4_950 native MPC: direct C++ integration

Updated by the user in this directory on 2026-10-31; used without modification:

- `nmpc.h`
  SHA256 `1bf9c1482288b642f454ccf53bfdba9ee5001a36af197097f81e2cfd94909d4c`
- `libnmpc.so.1`
  SHA256 `3ebd7f1c91ef7a42c2e69f66ac5c1679b33a1929bfe49669874af7d9323932e3`

The opt-in `sckd_native_mpc` target links this binary directly. It does not load
`libmpc_c_api.so`. CMake verifies the pair's hashes; the bridge checks the 832-byte
class size and rejects a runtime config pointing to another library. Rebuild and
revalidate ABI/control units before replacing this pair. The binary has no SONAME;
the executable records the absolute library path, so rebuild after moving the repo.

Build headers are bundled in `include/casadi` and `include/opencv4`. Native
runtime libraries are real copies in `runtime/` (not links to another checkout):
CasADi 3.7.2, its IPOPT plugin, IPOPT, OpenCV 4.6, Boost.Filesystem 1.83,
CoinHSL (MA57)/METIS, MUMPS/Scotch, BLAS/LAPACK/OpenBLAS and TBB. `dependencies.json` records original
sources and SHA256 hashes for every bundled library and header. Those source
paths are provenance only, not build or runtime dependencies.

The launcher sets `LD_LIBRARY_PATH` and `CASADIPATH` exclusively to this bundle
for the MPC process, preventing an inherited environment from selecting another
CasADi/OpenCV. The native contract test verifies process mappings contain local
CasADi/IPOPT and no IsaacLab or `/opt/openrobots` libraries. Original CasADi binary
RUNPATH metadata is retained; local search paths take priority and are verified.
System C/C++/Fortran/OpenMP, compression and X11/GL libraries remain Ubuntu 24.04
dependencies. The predictor is a separate process and still uses its existing
ball-mlp/OpenCV/TensorRT environment; this bundle covers native MPC only.

The 2026-10-31 binary selects MA57. `runtime/libcoinhsl.so` and its METIS dependency
are copies of the user's existing local installation, recorded in `dependencies.json`.
The relative alias `runtime/libhsl.so -> libcoinhsl.so` supplies the name loaded by
IPOPT. Keep these files together with the bundle; no system-wide installation or
solver override is needed. The contract test verifies MA57/METIS are loaded from
this project's runtime directory.

## Contract

State: `[x,y,q1..q5,vx,vy,dq1..dq5]`, target: `[x,y,z]`, three time segments in
seconds. Hit parameters are speed (m/s), yaw/pitch (degrees), vertical speed (m/s).
`opti_solution` returns the predicted terminal state; `get_controls` returns and
consumes 21 velocity increments in step-major order: `[dvx,dvy,dq1..dq5]` per
segment (m/s and rad/s). Starting from the measured input velocity, accumulate
`v_target[step]=v_initial+sum(delta_v[0..step])` without multiplying by segment
duration. Each accepted solve builds axis-major `plan_pos/plan_vel` from that
solve's measured state, using the nmpc6 1 ms velocity ramp and then position
integration. Acceleration limits are `[8,8,30,30,180,400,200]` (m/s² and rad/s²);
segment boundaries are truncated to milliseconds and plans are capped at 3 s.

Both native replay and nmpc6 issue the next millisecond's rounded trajectory
sample. Arm commands also include the configured command delay (zero for X4).
Native replay converts the sampled joint positions/velocities back through the
configured signs/offsets and sends them directly with X4 servo gains and gravity
feedforward. World-frame base velocities are rotated into body-frame twist with
zero yaw rate. The extra replay velocity/acceleration/position-window clamps are
removed; the X4 motor model and ChassisServo still apply their actuator constraints.

A new accepted solve replaces the trajectory and its time origin. Rejected solves
or invalid predictions leave the accepted trajectory running. After its endpoint,
arm positions extrapolate at terminal velocity for up to 150 ms, while the base
holds terminal velocity for that interval. Then all reference velocities become
zero and the arm holds the follow-through endpoint. Before any accepted solve,
the arm remains at standby and base twist is zero.

Each solve initializes both success flags to false and drains any previous
controls. Accept only a successful solver call with a nonempty finite terminal
state and 21 finite velocity increments. Preserve `accuracy_success` separately: the
library also accepts a relaxed position gate while this flag remains false.
`events.jsonl` records both flags, `control_semantics: "velocity_increment"`,
`velocity_increments`, segment duration, plan length and target velocities. A
`command` event on every 1 ms tick records the accepted plan origin, sample times
and indices, final joint position/velocity commands, and world/body base commands.
Reset destroys and reconstructs the controller. Use one controller serially;
the binary contains process-global state.

Geometry verified: d1=.90, a2=.706, d4=-.1705, racket_len=.865 (metres).
The X4_950 model now uses the same .865 m effective racket length, configured in
`models/config/x4_950.json`. Its Link6 Y offset is .930 m because Link5 contributes
-.065 m. The builder moves the face and center together; archived URDFs retain
their original dimensions. The remaining sub-millimetre/millimetre FK differences
come from the shoulder height and other link dimensions, not racket length.

## Verification

From the repository root:

```sh
cmake -S sckd_sim -B sckd_sim/build -DSCKD_BUILD_X4_NATIVE_REPLAY=ON
cmake --build sckd_sim/build --target x4_native_replay x4_shuttle_predictor test_native_mpc test_native_nmpc_alignment test_legacy_nmpc -j4
ctest --test-dir sckd_sim/build -R '^(native_mpc|native_nmpc_alignment|legacy_nmpc)$' --output-on-failure
.venv/bin/python -m unittest discover -s tests -p 'test_x4_native_execution.py' -v
python3 tools/run_x4_native_replay.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current --view
```

The contract test checks actual geometry, cumulative velocity increments against the
library's terminal velocity, consuming controls, dense trajectories, invalid/rejected
inputs, reset and destruction, and process mappings without the C API library.
The alignment test replays a successful nmpc6 solve's measured state and control
increments through the native trajectory builder, comparing every position and
velocity sample within `1e-10`, and covers millisecond boundaries and follow-through.
The execution test runs the real native replay with a controlled predictor socket,
checking next-tick commands, failed-solve/invalid-prediction plan retention, and
the 150 ms follow-through boundary against an independent trajectory calculation.
Headless runs omit `--view`. The default config retains the selected trajectory,
initial base position, prediction height and X4 actuator configuration.

实时窗口默认以 0.25 倍速运行（1 秒仿真时间约对应 4 秒墙钟时间），
可用 `--view-speed 0.1` 改为十分之一速度，或 `--view-speed 1` 恢复正常速度。
该参数只改变实时等待节奏，不改变物理时间步长和预测时间轴。
完整重置执行：关闭实时窗口，再次运行同一条 `--view` 命令；这会重新创建
仿真、预测器和 MPC，并从轨迹首点开始。窗口内目前没有整条控制链重启快捷键。
默认每次创建新的输出目录；如显式传入 `--output`，重新执行时需使用新的空目录。
