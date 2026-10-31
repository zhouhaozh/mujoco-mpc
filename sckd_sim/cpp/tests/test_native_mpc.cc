// Integration contract for the supplied X4 C++ ABI; run in a dedicated output directory.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "nmpc.h"
#include "control/native_mpc_bridge.h"

void Require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}

int main() try {
  Require(sizeof(NMPC) == 832, "ABI size mismatch");
  const std::array<double, 14> state = {0, 0, 0, .610849069, -.349214809, -1.570795754, 0,
                                      0, 0, 0, 0, -.000516937, 0, 0};
  const std::array<double, 3> target = {.395814209, -.030633183, .800788574};
  const double remaining = .394, dt = remaining/3;
  std::vector<double> expected_controls;
  {
    NMPC raw(3);
    const auto zero = raw.GetEndPos2(std::vector<double>(14, 0), {0,0,0}, false);
    Require(zero.size() >= 3 && std::abs(zero[0]-.706)<1e-10 &&
            std::abs(zero[1]-.865)<1e-10 && std::abs(zero[2]-1.0705)<1e-10,
            "new library geometry mismatch");
    bool success=false, accuracy=false;
    auto solution=raw.opti_solution({state.begin(),state.end()}, {target.begin(),target.end()},
        {dt,dt,dt}, success, accuracy, 4,0,10,.5,0);
    expected_controls=raw.get_controls();
    std::cout << "raw_solver=" << success << " accuracy=" << accuracy
              << " state_size=" << solution.size() << " controls=" << expected_controls.size() << '\n';
    Require(success && solution.size()>=14 && expected_controls.size()==21, "raw solve rejected");
    // Independent oracle: controls are per-segment velocity increments, not accelerations.
    bool differs_from_acceleration_interpretation = false;
    for(int axis=0;axis<7;++axis) {
      double velocity=state[7+axis], acceleration_interpretation=state[7+axis];
      for(int step=0;step<3;++step) {
        velocity+=expected_controls[step*7+axis];
        acceleration_interpretation+=expected_controls[step*7+axis]*dt;
      }
      Require(std::isfinite(velocity) && std::abs(velocity-solution[7+axis])<1e-8,
              "velocity increments do not match library terminal velocity");
      differs_from_acceleration_interpretation |=
          std::abs(acceleration_interpretation-solution[7+axis]) > 1e-4;
    }
    Require(differs_from_acceleration_interpretation,
            "test fixture does not distinguish velocity increments from accelerations");
    Require(raw.get_controls().empty(), "get_controls must consume its output");
  }
  {
    sckd::NativeMpcBridge bridge(SCKD_NATIVE_MPC_LIBRARY);
    bridge.SetHitParams({4,0,10,.5});
    const auto plan=bridge.Solve(state,target,remaining);
    Require(plan.success, plan.error.c_str());
    Require(plan.plan_ms==static_cast<int>(remaining*1000) &&
            plan.plan_pos.size()==7*static_cast<size_t>(plan.plan_ms+1) &&
            plan.plan_vel.size()==plan.plan_pos.size(),
            "bridge did not build a complete 1 ms trajectory");
    for(int axis=0;axis<7;++axis) {
      double velocity=state[7+axis];
      for(int step=0;step<3;++step) {
        const double increment=expected_controls[step*7+axis];
        velocity+=increment;
        Require(std::abs(increment-plan.velocity_increments[step][axis])<1e-5,
                "bridge changed a velocity increment");
        Require(std::abs(velocity-plan.velocities[step][axis])<1e-5,
                "bridge disagrees with independently accumulated velocity increments");
      }
      Require(std::abs(plan.plan_pos[axis*(plan.plan_ms+1)]-state[axis])<1e-12 &&
              std::abs(plan.plan_vel[axis*(plan.plan_ms+1)]-state[7+axis])<1e-12,
              "dense trajectory is not anchored at the measured state");
    }
    const auto next_tick=plan.Sample(.001);
    Require(next_tick.index==1, "next-tick execution sampled the measured-state point");
    Require(plan.Sample(std::numeric_limits<double>::quiet_NaN()).index==-1,
            "nonfinite plan time accepted");
    Require(!bridge.Solve(state,target,0).success, "zero time accepted");
    auto bad=state; bad[2]=std::numeric_limits<double>::quiet_NaN();
    Require(!bridge.Solve(bad,target,remaining).success, "NaN accepted");
    Require(!bridge.Solve(state,{1e6,1e6,1e6},remaining).success, "rejected target reused old plan");
    bridge.Reset();
    Require(bridge.Solve(state,target,remaining).success, "solve after reset failed");
    std::ifstream maps("/proc/self/maps");
    const std::string loaded((std::istreambuf_iterator<char>(maps)), {});
    Require(loaded.find("native_mpc_x4_950/libnmpc.so.1")!=std::string::npos, "new library not mapped");
    Require(loaded.find("libmpc_c_api.so")==std::string::npos, "old C API was loaded");
    Require(loaded.find("/opt/openrobots/")==std::string::npos &&
            loaded.find("/my_isaaclab_badminton/")==std::string::npos,
            "MPC dependencies were loaded from outside the project bundle");
    Require(loaded.find("native_mpc_x4_950/runtime/libcasadi.so.3.7")!=std::string::npos &&
            loaded.find("native_mpc_x4_950/runtime/libipopt.so.")!=std::string::npos,
            "project-local CasADi/IPOPT runtime was not used");
    Require(loaded.find("native_mpc_x4_950/runtime/libcoinhsl.so")!=std::string::npos &&
            loaded.find("native_mpc_x4_950/runtime/libmetis.so.")!=std::string::npos,
            "project-local MA57/METIS runtime was not used");
  }
  bool rejected=false;
  try { sckd::NativeMpcBridge wrong("/home/ubuntu/Downloads/libnmpc.so.1"); }
  catch(const std::invalid_argument&) { rejected=true; }
  Require(rejected, "different runtime library path accepted by fixed-ABI executable");
  std::cout << "native_cpp_contract_passed=true sizeof_NMPC=832 c_api_loaded=false\n";
  return 0;
} catch(const std::exception& e) {
  std::cerr << "native_cpp_contract_failed: " << e.what() << '\n';
  return 1;
}
