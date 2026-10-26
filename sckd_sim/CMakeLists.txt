cmake_minimum_required(VERSION 3.16)
project(sckd_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR "/home/ubuntu/research/sim/deps/mujoco-3.14.0" CACHE PATH
    "MuJoCo release root containing include/ and lib/")

# 旧仓库实机 NMPC 模块(legacy_nmpc):CasADi 3.7 运行库(nlpsol/ipopt/
# mumps 全套;实机部署指向 badminton-mpc-6.0/deps/casadi_portable 即可)
set(CASADI_DIR "/home/ubuntu/deps/casadi" CACHE PATH
    "CasADi 3.7 root containing include/ and lib/")

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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/apps/m1_chassis_step.cpp")
  add_executable(m1_chassis_step cpp/apps/m1_chassis_step.cpp)
  target_link_libraries(m1_chassis_step PRIVATE sckd_core)
endif()

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

add_executable(test_runtime_supervision cpp/tests/test_runtime_supervision.cc)
target_include_directories(test_runtime_supervision PRIVATE third_party/doctest)
target_link_libraries(test_runtime_supervision PRIVATE sckd_runtime_contracts)
add_test(NAME runtime_supervision COMMAND test_runtime_supervision)

# X4接入第一阶段：接口与日志无SDK依赖；SDK头仅进入独立翻译单元。
add_library(sckd_mujoco_robot_io STATIC cpp/runtime/mujoco_robot_backend.cpp)
target_link_libraries(sckd_mujoco_robot_io PUBLIC sckd_core sckd_runtime_contracts)
target_link_libraries(m2_intercept PRIVATE sckd_mujoco_robot_io)

add_executable(test_robot_io cpp/tests/test_robot_io.cc)
target_include_directories(test_robot_io PRIVATE cpp third_party/doctest)
add_test(NAME robot_io COMMAND test_robot_io)

add_executable(test_mujoco_robot_io cpp/tests/test_mujoco_robot_io.cc)
target_include_directories(test_mujoco_robot_io PRIVATE third_party/doctest)
target_link_libraries(test_mujoco_robot_io PRIVATE sckd_mujoco_robot_io)
add_test(NAME mujoco_robot_io COMMAND test_mujoco_robot_io WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(robot_io_replay cpp/apps/robot_io_replay.cpp)
target_link_libraries(robot_io_replay PRIVATE sckd_mujoco_robot_io)

set(X4_SDK_ROOT "" CACHE PATH "Optional X4 SDK release directory (headers and prebuilt libraries)")
if(X4_SDK_ROOT)
  foreach(component sdk chassis arm hal core)
    find_library(X4_${component}_LIB NAMES ${component} PATHS "${X4_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  endforeach()
  add_library(sckd_x4_robot_io STATIC cpp/runtime/x4_sdk_backend.cpp)
  # 两工程都有core/config.h，SDK编译必须优先选自身头；不向其他目标传播。
  target_include_directories(sckd_x4_robot_io BEFORE PRIVATE "${X4_SDK_ROOT}/include")
  target_include_directories(sckd_x4_robot_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
  target_link_libraries(sckd_x4_robot_io PRIVATE Eigen3::Eigen Threads::Threads
    ${X4_sdk_LIB} ${X4_chassis_LIB} ${X4_arm_LIB} ${X4_hal_LIB} ${X4_core_LIB})
  target_link_libraries(robot_io_replay PRIVATE sckd_x4_robot_io)
  target_compile_definitions(robot_io_replay PRIVATE SCKD_WITH_X4_SDK=1)
  set_property(TARGET robot_io_replay APPEND PROPERTY BUILD_RPATH "${X4_SDK_ROOT}/lib")
  add_executable(test_x4_sdk_backend cpp/tests/test_x4_sdk_backend.cc)
  target_include_directories(test_x4_sdk_backend PRIVATE third_party/doctest)
  target_link_libraries(test_x4_sdk_backend PRIVATE sckd_x4_robot_io)
  set_property(TARGET test_x4_sdk_backend APPEND PROPERTY BUILD_RPATH "${X4_SDK_ROOT}/lib")
  add_test(NAME x4_sdk_backend COMMAND test_x4_sdk_backend)
endif()
