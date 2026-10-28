cmake_minimum_required(VERSION 3.16)
project(sckd_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/mujoco" CACHE PATH
    "MuJoCo release root containing include/ and lib/")

# 旧仓库实机 NMPC 模块(legacy_nmpc):CasADi 运行库(nlpsol/ipopt/mumps)。
set(CASADI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/casadi" CACHE PATH
    "CasADi root containing include/ and lib/")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(sckd_core STATIC
  cpp/core/config.cpp
  cpp/robot/motor_model.cpp
  cpp/robot/swerve_kinematics.cpp
  cpp/hal/sim_backend.cpp
  cpp/control/chassis_servo.cpp
  cpp/control/chassis_mpc.cpp
  cpp/control/gen/chassis_mpc.c
  cpp/estimation/wheel_odom.cpp
  cpp/sim/ball_aero.cpp
  cpp/sim/ball_server.cpp
  cpp/estimation/ball_predictor.cpp
  cpp/planning/hit_planner.cpp  cpp/control/arm_ik.cpp
  cpp/control/swing_generator.cpp cpp/control/arm_swing.cpp
  cpp/control/wholebody_ocp.cpp cpp/control/wholebody_mpc.cpp)
set_source_files_properties(cpp/control/gen/chassis_mpc.c PROPERTIES LANGUAGE CXX)
target_include_directories(sckd_core PUBLIC third_party/osqp/include)
target_link_libraries(sckd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osqp/libosqp.a)
target_include_directories(sckd_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${MUJOCO_DIR}/include)
target_link_directories(sckd_core PUBLIC ${MUJOCO_DIR}/lib)
target_link_libraries(sckd_core PUBLIC mujoco Threads::Threads Eigen3::Eigen)

foreach(app m0_determinism m0_step_rate m0_static_contact m0_arm_step_response
        m1_traj_tracking m1_arm_gravity m1_odom m1_dbg_chassis m1_step_resp m1_mpc_tracking m2_predictor m2_intercept m2_demo demo_hit)
  add_executable(${app} cpp/apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE sckd_core)
endforeach()

# Measured initial state + recorded command replay through the existing X4
# actuator, chassis-servo and MuJoCo dynamics stack. No planner/MPC dependency.
add_executable(x4_command_replay cpp/apps/x4_command_replay.cpp)
target_link_libraries(x4_command_replay PRIVATE sckd_core)

# MIT 协议单机仿真(2026-10-18):纯标准库独立可执行,不链 sckd_core/MuJoCo,
# 复现真机臂执行链(4×16bit 量化帧/1kHz 驱动器/9ms 延迟/RS04 T-N 包络)
add_executable(mit_motor_sim cpp/apps/mit_motor_sim.cpp)

# 5关节臂 MIT 前馈辨识仿真(2026-10-20):直链 MuJoCo(耦合真值对象),
# 不链 sckd_core(免 Eigen/OSQP);量化帧/延迟/T-N 执行链与 mit_motor_sim 同构
add_executable(mit_arm_id_sim cpp/apps/mit_arm_id_sim.cpp)
target_include_directories(mit_arm_id_sim PRIVATE ${MUJOCO_DIR}/include)
target_link_directories(mit_arm_id_sim PRIVATE ${MUJOCO_DIR}/lib)
target_link_libraries(mit_arm_id_sim PRIVATE mujoco Threads::Threads)

# m2_intercept:argv[6]="nmpc6" 模式(旧仓库实机 NMPC A/B 接线,2026-10-09)
target_link_libraries(m2_intercept PRIVATE sckd_nmpc6)

add_executable(test_motor_model cpp/tests/test_motor_model.cpp)
target_include_directories(test_motor_model PRIVATE third_party/doctest)
target_link_libraries(test_motor_model PRIVATE sckd_core)

add_executable(test_swerve cpp/tests/test_swerve.cc)
target_include_directories(test_swerve PRIVATE third_party/doctest)
target_link_libraries(test_swerve PRIVATE sckd_core)

add_executable(test_ball cpp/tests/test_ball.cc)
target_include_directories(test_ball PRIVATE third_party/doctest)
target_link_libraries(test_ball PRIVATE sckd_core)

add_executable(test_wbo cpp/tests/test_wbo.cc)
target_include_directories(test_wbo PRIVATE third_party/doctest)
target_link_libraries(test_wbo PRIVATE sckd_core)

add_executable(test_wbmpc cpp/tests/test_wbmpc.cc)
target_include_directories(test_wbmpc PRIVATE third_party/doctest)
target_link_libraries(test_wbmpc PRIVATE sckd_core)

# 旧仓库实机 NMPC 独立模块(2026-10-09 移植):静态库隔离 casadi 依赖,
# 不上 sckd_core(m0/m1 系 app 不背运行库);需要者显式链接
add_library(sckd_nmpc6 STATIC cpp/control/legacy_nmpc.cpp)
target_include_directories(sckd_nmpc6 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nmpc6
  ${CASADI_DIR}/include)
target_link_directories(sckd_nmpc6 PUBLIC ${CASADI_DIR}/lib)
target_link_libraries(sckd_nmpc6 PUBLIC casadi ${CMAKE_DL_LIBS})
set_target_properties(sckd_nmpc6 PROPERTIES
  BUILD_RPATH "${CASADI_DIR}/lib" INSTALL_RPATH "${CASADI_DIR}/lib")

add_executable(test_legacy_nmpc cpp/tests/test_legacy_nmpc.cc)
target_include_directories(test_legacy_nmpc PRIVATE third_party/doctest)
target_link_libraries(test_legacy_nmpc PRIVATE sckd_nmpc6)

enable_testing()
add_test(NAME motor_model COMMAND test_motor_model)
add_test(NAME swerve COMMAND test_swerve)
add_test(NAME ball COMMAND test_ball)
add_test(NAME wbo COMMAND test_wbo WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME wbmpc COMMAND test_wbmpc WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME legacy_nmpc COMMAND test_legacy_nmpc
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Read-only external predictor contract probe; no motor/simulator commands.
add_executable(m2_prediction_probe cpp/apps/m2_prediction_probe.cpp)
target_link_libraries(m2_prediction_probe PRIVATE sckd_core)

# 原生实测采样时钟和显示参考必须分离，检查未来前缀不影响已收到帧。
add_executable(measured_replay_check cpp/apps/measured_replay_check.cpp)
target_include_directories(measured_replay_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
add_test(NAME measured_replay COMMAND measured_replay_check ${CMAKE_CURRENT_BINARY_DIR}/measured-check)

add_executable(m1_arm_identification cpp/apps/m1_arm_identification.cpp)
target_link_libraries(m1_arm_identification PRIVATE sckd_core)

add_executable(m1_chassis_identification cpp/apps/m1_chassis_identification.cpp)
target_link_libraries(m1_chassis_identification PRIVATE sckd_core)

add_executable(test_steer_rate_limits cpp/tests/test_steer_rate_limits.cc)
target_include_directories(test_steer_rate_limits PRIVATE third_party/doctest)
target_link_libraries(test_steer_rate_limits PRIVATE sckd_core)
add_test(NAME steer_rate_limits COMMAND test_steer_rate_limits)

# 同步pipeline边界。契约/配置不携带仿真或solver链接依赖；具体运行适配分层链接。
add_library(sckd_runtime_contracts INTERFACE)
target_include_directories(sckd_runtime_contracts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

add_library(sckd_runtime_execution STATIC cpp/runtime/trajectory_executor.cpp)
target_link_libraries(sckd_runtime_execution PUBLIC sckd_core sckd_runtime_contracts)
add_library(sckd_runtime_planning STATIC
  cpp/runtime/controller_adapters.cpp cpp/runtime/planning_session.cpp)
target_link_libraries(sckd_runtime_planning PUBLIC sckd_core sckd_nmpc6 sckd_runtime_contracts)
add_library(sckd_runtime_telemetry STATIC
  cpp/runtime/control_telemetry.cpp cpp/runtime/video_trace.cpp)
target_link_libraries(sckd_runtime_telemetry PUBLIC sckd_core sckd_runtime_contracts)
target_include_directories(sckd_runtime_telemetry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nmpc6)
add_library(sckd_runtime_simulation STATIC
  cpp/runtime/state_inputs.cpp cpp/runtime/simulation_step.cpp cpp/runtime/simulation_physics.cpp)
target_link_libraries(sckd_runtime_simulation PUBLIC sckd_core sckd_runtime_contracts)
add_library(sckd_runtime INTERFACE)
target_link_libraries(sckd_runtime INTERFACE sckd_runtime_planning sckd_runtime_execution
  sckd_runtime_telemetry sckd_runtime_simulation)
target_link_libraries(m2_intercept PRIVATE sckd_runtime)

add_executable(test_runtime_config cpp/tests/test_runtime_config.cc)
target_include_directories(test_runtime_config PRIVATE third_party/doctest)
target_link_libraries(test_runtime_config PRIVATE sckd_runtime_contracts)
add_test(NAME runtime_config COMMAND test_runtime_config)

# 仅配置/静态模型检查；不使能电机、不调用控制器或 mj_step。
add_executable(test_x4_950_config cpp/tests/test_x4_950_config.cc)
target_include_directories(test_x4_950_config PRIVATE third_party/doctest)
target_link_libraries(test_x4_950_config PRIVATE sckd_core sckd_runtime_contracts)
add_test(NAME x4_950_config COMMAND test_x4_950_config WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_runtime_supervision cpp/tests/test_runtime_supervision.cc)
target_include_directories(test_runtime_supervision PRIVATE third_party/doctest)
target_link_libraries(test_runtime_supervision PRIVATE sckd_runtime_contracts)
add_test(NAME runtime_supervision COMMAND test_runtime_supervision)

# Opt-in native X4 replay. Keep the predictor's OpenCV/TensorRT runtime in a
# separate process from the supplied MPC library's OpenCV/CasADi runtime.
option(SCKD_BUILD_X4_NATIVE_REPLAY "Build supplied-library X4 measured replay" OFF)
if(SCKD_BUILD_X4_NATIVE_REPLAY)
  set(BALL_MLP_ROOT "/home/ubuntu/works/ball-mlp" CACHE PATH "Existing ball-mlp checkout and build")
  find_library(BALL_MLP_LIBRARY NAMES ball-mlp PATHS "${BALL_MLP_ROOT}/build/src/ball-mlp" NO_DEFAULT_PATH REQUIRED)
  find_library(BALL_MLP_OPENCV NAMES opencv_core PATHS /usr/local/lib NO_DEFAULT_PATH REQUIRED)
  find_library(BALL_MLP_CUDEV NAMES opencv_cudev PATHS /usr/local/lib NO_DEFAULT_PATH REQUIRED)
  find_library(BALL_MLP_FMT NAMES fmt REQUIRED)
  find_library(BALL_MLP_CUDART NAMES cudart PATHS /usr/local/cuda/lib64 REQUIRED)
  find_library(BALL_MLP_TENSORRT NAMES nvinfer PATHS /usr/local/TensorRT/lib REQUIRED)
  add_executable(x4_shuttle_predictor cpp/apps/x4_shuttle_predictor.cpp cpp/estimation/measured_shuttle_predictor.cpp)
  target_include_directories(x4_shuttle_predictor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
  target_include_directories(x4_shuttle_predictor SYSTEM PRIVATE ${BALL_MLP_ROOT}/include)
  target_link_libraries(x4_shuttle_predictor PRIVATE ${BALL_MLP_LIBRARY} ${BALL_MLP_OPENCV}
    ${BALL_MLP_CUDEV} ${BALL_MLP_FMT} ${BALL_MLP_CUDART} ${BALL_MLP_TENSORRT} ${CMAKE_DL_LIBS} Threads::Threads rt)
  set(NATIVE_MPC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/native_mpc_x4_950")
  set(NATIVE_MPC_LIBRARY "${NATIVE_MPC_DIR}/libnmpc.so.1")
  file(SHA256 "${NATIVE_MPC_LIBRARY}" NATIVE_MPC_SHA256)
  file(SHA256 "${NATIVE_MPC_DIR}/nmpc.h" NATIVE_MPC_HEADER_SHA256)
  if(NOT NATIVE_MPC_SHA256 STREQUAL "2afcda72d20e7a0f77798d101049aba97cb7d9952b10da1ac6b82aaea9607554"
     OR NOT NATIVE_MPC_HEADER_SHA256 STREQUAL "e3b82039f985691e37c42e1638404d62a2bbbeef3f5e0c11a7df39f3a8d35389")
    message(FATAL_ERROR "Native MPC header/library pair changed; verify ABI and control units before updating hashes")
  endif()
  set(NATIVE_MPC_DEPS "${NATIVE_MPC_DIR}/runtime")
  set(NATIVE_MPC_OPENCV_INCLUDE "${NATIVE_MPC_DIR}/include/opencv4")
  add_library(x4_nmpc_binary SHARED IMPORTED)
  set_target_properties(x4_nmpc_binary PROPERTIES IMPORTED_LOCATION "${NATIVE_MPC_LIBRARY}")
  add_library(sckd_native_mpc STATIC cpp/control/native_mpc_bridge.cpp)
  target_include_directories(sckd_native_mpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
  target_include_directories(sckd_native_mpc SYSTEM PRIVATE
      ${NATIVE_MPC_DIR} ${NATIVE_MPC_DIR}/include ${NATIVE_MPC_OPENCV_INCLUDE})
  target_compile_definitions(sckd_native_mpc PRIVATE SCKD_NATIVE_MPC_LIBRARY="${NATIVE_MPC_LIBRARY}")
  target_link_libraries(sckd_native_mpc PUBLIC x4_nmpc_binary "${NATIVE_MPC_DEPS}/libcasadi.so.3.7" Threads::Threads)
  target_link_options(sckd_native_mpc INTERFACE "-Wl,-rpath-link,${NATIVE_MPC_DEPS}")
  add_executable(x4_native_replay cpp/apps/x4_native_replay.cpp)
  target_link_libraries(x4_native_replay PRIVATE sckd_core sckd_native_mpc)
  add_executable(test_native_mpc cpp/tests/test_native_mpc.cc)
  target_link_libraries(test_native_mpc PRIVATE sckd_native_mpc)
  target_include_directories(test_native_mpc SYSTEM PRIVATE
      ${NATIVE_MPC_DIR} ${NATIVE_MPC_DIR}/include ${NATIVE_MPC_OPENCV_INCLUDE})
  target_compile_definitions(test_native_mpc PRIVATE SCKD_NATIVE_MPC_LIBRARY="${NATIVE_MPC_LIBRARY}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/native-mpc-check")
  add_test(NAME native_mpc COMMAND test_native_mpc)
  set_tests_properties(native_mpc PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/native-mpc-check"
      ENVIRONMENT "LD_LIBRARY_PATH=${NATIVE_MPC_DEPS};CASADIPATH=${NATIVE_MPC_DEPS};OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
      TIMEOUT 60)
endif()
