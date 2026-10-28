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
  Require(sizeof(NMPC) == 856, "ABI size mismatch");
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
    // Independent oracle: the library's own predicted final state must agree with a*dt.
    for(int axis=0;axis<7;++axis) {
      double velocity=state[7+axis];
      for(int step=0;step<3;++step) velocity+=expected_controls[step*7+axis]*dt;
      Require(std::isfinite(velocity) && std::abs(velocity-solution[7+axis])<1e-8,
              "controls do not integrate to library terminal velocity");
    }
    Require(raw.get_controls().empty(), "get_controls must consume its output");
  }
  {
    sckd::NativeMpcBridge bridge(SCKD_NATIVE_MPC_LIBRARY);
    bridge.SetHitParams({4,0,10,.5});
    const auto plan=bridge.Solve(state,target,remaining);
    Require(plan.success, plan.error.c_str());
    for(int axis=0;axis<7;++axis) {
      double velocity=state[7+axis];
      double integrated_position=state[axis];
      double expected_position=state[axis];
      for(int step=0;step<3;++step) {
        const double initial_velocity=velocity;
        velocity+=expected_controls[step*7+axis]*dt;
        Require(std::abs(velocity-plan.velocities[step][axis])<1e-5,
                "bridge disagrees with independently solved acceleration plan");
        const auto start=plan.VelocityAt(step*dt);
        const auto midpoint=plan.VelocityAt((step+.5)*dt);
        const auto end=plan.VelocityAt((step+1)*dt);
        Require(std::abs(start[axis]-initial_velocity)<1e-5 &&
                std::abs(midpoint[axis]-(initial_velocity+velocity)*.5)<1e-5 &&
                std::abs(end[axis]-velocity)<1e-5, "segment velocity is applied ahead of time");
        integrated_position+=midpoint[axis]*dt;
        expected_position+=initial_velocity*dt+.5*expected_controls[step*7+axis]*dt*dt;
      }
      Require(std::abs(integrated_position-expected_position)<1e-5,
              "executed velocity does not integrate to constant-acceleration displacement");
    }
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
  }
  bool rejected=false;
  try { sckd::NativeMpcBridge wrong("/home/ubuntu/Downloads/libnmpc.so.1"); }
  catch(const std::invalid_argument&) { rejected=true; }
  Require(rejected, "different runtime library path accepted by fixed-ABI executable");
  std::cout << "native_cpp_contract_passed=true sizeof_NMPC=856 c_api_loaded=false\n";
  return 0;
} catch(const std::exception& e) {
  std::cerr << "native_cpp_contract_failed: " << e.what() << '\n';
  return 1;
}
