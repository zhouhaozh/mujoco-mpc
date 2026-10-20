// 旧仓库 NMPC 独立模块单测(2026-10-09 移植调通门槛):
// 用例参数逐位取自 badminton-mpc-6.0/test/test_mpc.cc 的实机日志串 ——
// 该参数在原机上有已知成功解,本测试即"调通"判据(同输入 → 可执行解)。
// 运行目录须为 sckd_sim/(nlp.so 相对路径)。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "control/legacy_nmpc.h"

using namespace sckd;

namespace {

// 实机日志参数(test_mpc.cc parameters_str 前段)
constexpr double kState[14] = {2.78,       0.03,   -0.0193708, 0.276508,
                               -0.34669,   -1.5647, 0.0161073,  0.503533,
                               -0.0818887, 0,      0,          0,
                               0,          0};
constexpr double kTarget[3] = {3.59, -1.26, 0.8};
constexpr double kDt = 0.158679;   // 剩余 0.476s / 3 步
constexpr double kHitSpd = 5.0, kHitZ = 2.1, kHitPitch = 61.0, kHitYaw = 0.0;

LegacyNmpcRequest MakeReq() {
  LegacyNmpcRequest r;
  for (int i = 0; i < 14; ++i) r.state[i] = kState[i];
  for (int i = 0; i < 3; ++i) r.target[i] = kTarget[i];
  r.dt[0] = r.dt[1] = r.dt[2] = kDt;
  r.hit_spd = kHitSpd;
  r.hit_speed_z = kHitZ;
  r.hit_pitch_deg = kHitPitch;
  r.hit_yaw_deg = kHitYaw;
  return r;
}

const char* StatusName(LegacyNmpcStatus s) {
  switch (s) {
    case LegacyNmpcStatus::kOk: return "ok";
    case LegacyNmpcStatus::kLoose: return "loose";
    case LegacyNmpcStatus::kSolverFail: return "solver-fail";
    case LegacyNmpcStatus::kSelfCollision: return "self-collision";
    case LegacyNmpcStatus::kJump: return "jump";
    case LegacyNmpcStatus::kAngleLimit: return "angle-limit";
    case LegacyNmpcStatus::kAccFail: return "acc-fail";
  }
  return "?";
}

}  // namespace

TEST_CASE("nonuniform capability cannot be enabled for the default legacy generated library") {
  LegacyNmpcConfig config;
  config.allow_nonuniform_dt = true;
  CHECK_THROWS_AS(LegacyNmpc{config}, std::runtime_error);
}

TEST_CASE("legacy single-dt generated library rejects nonuniform or invalid time requests") {
  LegacyNmpc nmpc;
  auto request = MakeReq();
  request.dt[1] *= .5;
  LegacyNmpcResult result;
  CHECK_FALSE(nmpc.Solve(request, &result));
  CHECK(result.status == LegacyNmpcStatus::kInputInvalid);
  CHECK(result.ipopt_status == "not_called");
  CHECK(result.plan_ms == 0);
  request = MakeReq();
  request.dt[2] = 0;
  CHECK_FALSE(nmpc.Solve(request, &result));
  CHECK(result.status == LegacyNmpcStatus::kInputInvalid);
  CHECK(result.ipopt_status == "not_called");
}

TEST_CASE("legacy_nmpc: 实机日志参数回归(调通门槛)") {
  LegacyNmpc nmpc;
  LegacyNmpcResult r;
  const LegacyNmpcRequest req = MakeReq();
  const bool ok = nmpc.Solve(req, &r);
  std::printf("[nmpc6-test] status=%s solver_ok=%d e=(%.3f,%.3f,%.3f) "
              "vx=%.2f ms=%.1f\n",
              StatusName(r.status), (int)r.solver_ok,
              std::fabs(r.end_pos[0] - req.target[0]),
              std::fabs(r.end_pos[1] - req.target[1]),
              std::fabs(r.end_pos[2] - req.target[2]),
              r.end_pos[4], r.solve_ms);
  REQUIRE(ok);
  CHECK((r.status == LegacyNmpcStatus::kOk ||
         r.status == LegacyNmpcStatus::kLoose));
  CHECK(r.solver_ok);
  // 精度门语义:接受解三轴误差必 <0.20m(宽松档下限)
  CHECK(std::fabs(r.end_pos[0] - req.target[0]) < 0.20);
  CHECK(std::fabs(r.end_pos[1] - req.target[1]) < 0.20);
  CHECK(std::fabs(r.end_pos[2] - req.target[2]) < 0.20);
  // 严格档还要求拍速 |vx|>0.6·hit_spd=3.0(本用例 pitch=61/z<0.9=前场球)
  if (r.status == LegacyNmpcStatus::kOk)
    CHECK(std::fabs(r.end_pos[4]) > 0.6 * kHitSpd);
  // 首解预算(冷解,研究口径;原机日志几十 ms 量级,CI 放宽)
  CHECK(r.solve_ms < 2000.0);
  // 末端 z 物理合法(自碰线以上)
  CHECK(r.end_pos[2] > 0.35);
  // 触球底盘速度已捕获
  CHECK(r.pred_vxy_at_hit[0] == r.pred_vxy_at_hit[0]);   // 有限
}

TEST_CASE("legacy_nmpc: 拍间重解(dt 收缩)+ 回合 Reset") {
  LegacyNmpc nmpc;
  LegacyNmpcResult r1, r2, r3;
  REQUIRE(nmpc.Solve(MakeReq(), &r1));

  // 60ms 后重解:剩余 0.416s(拍间 dt 收缩;状态近似不动 —— 守门链走热路径)
  LegacyNmpcRequest req2 = MakeReq();
  for (int i = 0; i < 3; ++i) req2.dt[i] = (3 * kDt - 0.06) / 3;
  const bool ok2 = nmpc.Solve(req2, &r2);
  std::printf("[nmpc6-test] replan status=%s ms=%.1f\n",
              StatusName(r2.status), r2.solve_ms);
  // 重解可能因 dt 缩短被精度门拒(时间不够)—— 只要求守门链优雅返回:
  // 不崩、状态显式、拒绝时不产出可执行解
  if (!ok2) {
    CHECK(r2.status != LegacyNmpcStatus::kOk);
    CHECK(r2.status != LegacyNmpcStatus::kLoose);
  } else {
    CHECK(std::fabs(r2.end_pos[2] - req2.target[2]) < 0.25);
  }

  // 回合 Reset 后再解:回到冷路径且可解(全零 guess 自动回合重置同源)
  nmpc.Reset();
  REQUIRE(nmpc.Solve(MakeReq(), &r3));
  CHECK(r3.status == r1.status);   // 同输入同回合起点 → 同档
  CHECK(std::fabs(r3.term_state[2] - r1.term_state[2]) < 1e-9);
}

TEST_CASE("legacy_nmpc: 不可达目标 → 守门显式拒绝(不产出可执行解)") {
  LegacyNmpc nmpc;
  LegacyNmpcResult r;
  LegacyNmpcRequest req = MakeReq();
  // 剩余 0.09s 内到 x+1.5m:任何守门(自碰/超窗/精度)拒绝都合法,
  // 断言仅"返回 false 且原因显式"
  req.target[0] = req.state[0] + 1.5;
  req.dt[0] = req.dt[1] = req.dt[2] = 0.03;
  const bool ok = nmpc.Solve(req, &r);
  if (!ok) {
    CHECK(r.status != LegacyNmpcStatus::kOk);
    CHECK(r.status != LegacyNmpcStatus::kLoose);
  } else {
    // 若居然可解(宽松档),误差也必须过宽松门 —— 门语义自洽
    CHECK(std::fabs(r.end_pos[0] - req.target[0]) < 0.20);
  }
}

TEST_CASE("simulation negative J2 standby uses MJCF frame bounds") {
  // Verify a negative-J2 starting pose can produce an accepted solution
  // while terminal J2/J3 stay inside the loaded simulator joint ranges.
  LegacyNmpcRequest req=MakeReq();
  req.state[3]=-0.35;
  req.dt[0]=req.dt[1]=req.dt[2]=0.32;
  LegacyNmpcConfig cfg; cfg.sim_joint23_limits=true;
  LegacyNmpc adapted(cfg);
  LegacyNmpcResult sim;
  const bool ok=adapted.Solve(req,&sim);
  REQUIRE(sim.solver_ok);
  REQUIRE(ok);
  CHECK(sim.term_state[3]>=cfg.joint23_lo[0]);
  CHECK(sim.term_state[3]<=cfg.joint23_hi[0]);
  CHECK(sim.term_state[4]>=cfg.joint23_lo[1]);
  CHECK(sim.term_state[4]<=cfg.joint23_hi[1]);
}

TEST_CASE("corrected request rejects invalid clocks and nonfinite state") {
  LegacyNmpcConfig cfg;cfg.enforce_feasibility=true;LegacyNmpc n(cfg);LegacyNmpcResult r;
  auto q=MakeReq();q.dt[1]=0;CHECK_FALSE(n.Solve(q,&r));CHECK(r.status==LegacyNmpcStatus::kInputInvalid);
  q=MakeReq();q.state[0]=NAN;CHECK_FALSE(n.Solve(q,&r));CHECK(r.status==LegacyNmpcStatus::kInputInvalid);
}
TEST_CASE("terminal anchor does not constrain first intermediate node") {
  LegacyNmpcConfig cfg;cfg.enforce_feasibility=true;cfg.sim_joint23_limits=true;
  LegacyNmpc n(cfg);LegacyNmpcResult r;auto q=MakeReq();
  q.dt[0]=q.dt[1]=q.dt[2]=.32;q.state[3]=-.35;
  const bool ok=n.Solve(q,&r);
  if(ok){CHECK(r.solver_ok);CHECK(r.max_constraint_violation<=cfg.constraint_tolerance);CHECK(std::hypot(r.end_pos[0]-q.target[0],r.end_pos[1]-q.target[1],r.end_pos[2]-q.target[2])<=cfg.terminal_tolerance_m);}
  q.base_anchor_ok=true;q.base_anchor[0]=5.5;q.base_anchor[1]=1.8;q.base_anchor_r=.02;q.dt[0]=q.dt[1]=q.dt[2]=.02;
  CHECK_FALSE(n.Solve(q,&r));CHECK_FALSE((r.status==LegacyNmpcStatus::kOk||r.status==LegacyNmpcStatus::kLoose));
}
