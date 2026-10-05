cmake_minimum_required(VERSION 3.16)
project(sckd_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR "/home/ubuntu/research/sim/deps/mujoco-3.14.0" CACHE PATH
    "MuJoCo release root containing include/ and lib/")

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
  cpp/control/swing_generator.cpp)
set_source_files_properties(cpp/control/gen/chassis_mpc.c PROPERTIES LANGUAGE CXX)
target_include_directories(sckd_core PUBLIC third_party/osqp/include)
target_link_libraries(sckd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osqp/libosqp.a)
target_include_directories(sckd_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${MUJOCO_DIR}/include)
target_link_directories(sckd_core PUBLIC ${MUJOCO_DIR}/lib)
target_link_libraries(sckd_core PUBLIC mujoco Threads::Threads Eigen3::Eigen)

foreach(app m0_determinism m0_step_rate m0_static_contact m0_arm_step_response
        m1_traj_tracking m1_arm_gravity m1_odom m1_dbg_chassis m1_mpc_tracking m2_predictor m2_intercept demo_hit)
  add_executable(${app} cpp/apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE sckd_core)
endforeach()

add_executable(test_motor_model cpp/tests/test_motor_model.cpp)
target_include_directories(test_motor_model PRIVATE third_party/doctest)
target_link_libraries(test_motor_model PRIVATE sckd_core)

add_executable(test_swerve cpp/tests/test_swerve.cc)
target_include_directories(test_swerve PRIVATE third_party/doctest)
target_link_libraries(test_swerve PRIVATE sckd_core)

add_executable(test_ball cpp/tests/test_ball.cc)
target_include_directories(test_ball PRIVATE third_party/doctest)
target_link_libraries(test_ball PRIVATE sckd_core)

enable_testing()
add_test(NAME motor_model COMMAND test_motor_model)
add_test(NAME swerve COMMAND test_swerve)
add_test(NAME ball COMMAND test_ball)
