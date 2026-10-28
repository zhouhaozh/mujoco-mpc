# X4_950 native MPC: direct C++ integration

Supplied by the user on 2026-10-28; copied without modification:

- `nmpc.h`: `/home/ubuntu/Downloads/nmpc.h`
  SHA256 `e3b82039f985691e37c42e1638404d62a2bbbeef3f5e0c11a7df39f3a8d35389`
- `libnmpc.so.1`: `/home/ubuntu/Downloads/libnmpc.so.1`
  SHA256 `2afcda72d20e7a0f77798d101049aba97cb7d9952b10da1ac6b82aaea9607554`

The opt-in `sckd_native_mpc` target links this binary directly. It does not load
`libmpc_c_api.so`. CMake verifies the pair's hashes; the bridge checks the 856-byte
class size and rejects a runtime config pointing to another library. Rebuild and
revalidate ABI/control units before replacing this pair. The binary has no SONAME;
the executable records the absolute library path, so rebuild after moving the repo.

Build headers are bundled in `include/casadi` and `include/opencv4`. Native
runtime libraries are real copies in `runtime/` (not links to another checkout):
CasADi 3.7.2, its IPOPT plugin, IPOPT, OpenCV 4.6, Boost.Filesystem 1.83,
MUMPS/Scotch, BLAS/LAPACK/OpenBLAS and TBB. `dependencies.json` records original
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

## Contract

State: `[x,y,q1..q5,vx,vy,dq1..dq5]`, target: `[x,y,z]`, three time segments in
seconds. Hit parameters are speed (m/s), yaw/pitch (degrees), vertical speed (m/s).
`opti_solution` returns the predicted terminal state; `get_controls` returns and
consumes 21 accelerations in step-major order. Convert with `v_next=v+a*dt`, not
`v_next=v+a`. Execution evaluates `v_start+a*t` within each segment, then applies the existing
velocity, acceleration and servo limits.

Each solve initializes both success flags to false and drains any previous
controls. Accept only a successful solver call with a nonempty finite terminal
state and 21 finite accelerations. Preserve `accuracy_success` separately: the
library also accepts a relaxed position gate while this flag remains false.
`events.jsonl` records both flags, accelerations, segment duration and velocities.
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
cmake --build sckd_sim/build --target x4_native_replay x4_shuttle_predictor test_native_mpc -j4
ctest --test-dir sckd_sim/build -R '^native_mpc$' --output-on-failure
python3 tools/run_x4_native_replay.py --view
```

The contract test checks actual geometry, acceleration integration against the
library's terminal velocity, consuming controls, bridge output, invalid/rejected
inputs, reset and destruction, and process mappings without the C API library.
Headless runs omit `--view`. The default config retains the selected trajectory,
initial base position, prediction height and motion limits.

实时窗口默认以 0.25 倍速运行（1 秒仿真时间约对应 4 秒墙钟时间），
可用 `--view-speed 0.1` 改为十分之一速度，或 `--view-speed 1` 恢复正常速度。
该参数只改变实时等待节奏，不改变物理时间步长和预测时间轴。
完整重置执行：关闭实时窗口，再次运行同一条 `--view` 命令；这会重新创建
仿真、预测器和 MPC，并从轨迹首点开始。窗口内目前没有整条控制链重启快捷键。
默认每次创建新的输出目录；如显式传入 `--output`，重新执行时需使用新的空目录。
