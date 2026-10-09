// 全身 MPC 联合 SQP 单元测试(2026-10-09):python wholebody_ocp.py v2 的
// A 跑(联合自由)参考真值,断言 C++ SQP 收敛到同一联合最优邻域。
// 参考真值(python IPOPT,38-48ms/解):
//   G1: term_base=(0.66,-0.48) |sp|=0 peak|dq|=4.6 n·n*=1.000
//   G2: term_base=(0.44,-0.28) |sp|=0 peak|dq|=6.3
//   G3: term_base=(0.78,-0.66) |sp|=0 peak|dq|=5.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>

#include <mujoco/mujoco.h>

#include "control/arm_ik.h"
#include "control/wholebody_mpc.h"

using namespace sckd;

namespace {

constexpr const char* kModel = "models/generated/x4_tennis_demo.mjcf";

void ReboundInverse(const double v_in[3], const double v_out[3],
                    double v_star[3], double n_star[3]) {
  const double vo = std::sqrt(v_out[0] * v_out[0] + v_out[1] * v_out[1] +
                              v_out[2] * v_out[2]);
  for (int i = 0; i < 3; ++i) n_star[i] = v_out[i] / std::max(vo, 0.01);
  const double v_in_n = v_in[0] * n_star[0] + v_in[1] * n_star[1] + v_in[2] * n_star[2];
  const double vr_n = (vo - (-0.5) * v_in_n) / 1.5;
  for (int i = 0; i < 3; ++i) v_star[i] = n_star[i] * vr_n;
}

struct Fixture {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  ArmIk* ik = nullptr;
  Fixture() {
    char err[512] = "";
    m = mj_loadXML(kModel, nullptr, err, sizeof(err));
    REQUIRE_MESSAGE(m != nullptr, err);
    d = mj_makeData(m);
    mj_resetDataKeyframe(m, d, 0);
    ik = new ArmIk(m, d);
  }
  ~Fixture() {
    delete ik;
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
  }
};

struct Scenario {
  const char* name;
  double p_hit[3], v_in[3], T;
  double base0[2], vbase0[2];
  double ref_base[2];   // python A 跑参考
};

const Scenario kScen[] = {
    {"G1", {0.95, -0.55, 0.70}, {-5.5, -0.3, -1.7}, 0.85, {-0.05, -0.10}, {0.8, -0.4}, {0.66, -0.48}},
    {"G2", {0.70, 0.90, 0.62}, {-5.8, 0.2, -1.5}, 0.95, {0.05, 0.30}, {0.2, -1.0}, {0.44, -0.28}},
    {"G3", {1.30, -0.85, 0.75}, {-5.2, -0.5, -1.8}, 0.80, {0.0, -0.20}, {1.1, -0.5}, {0.78, -0.66}},
};

}  // namespace

TEST_CASE("wholebody_mpc: G1-G3 收敛到 python 联合最优邻域") {
  Fixture fx;
  for (const auto& sc : kScen) {
    CAPTURE(sc.name);
    WholebodyMpc mpc(*fx.ik, fx.m, WmLimits());
    WmRequest req;
    req.t_now = 1.00;
    req.t_hit = 1.00 + sc.T;
    for (int i = 0; i < 3; ++i) req.p_hit[i] = sc.p_hit[i];
    const double el = 15.0 * M_PI / 180.0;
    const double v_out[3] = {8.0 * std::cos(el), 0.0, 8.0 * std::sin(el)};
    double n_star[3];
    ReboundInverse(sc.v_in, v_out, req.v_star, n_star);
    for (int i = 0; i < 2; ++i) {
      req.base0[i] = sc.base0[i];
      req.vbase0[i] = sc.vbase0[i];
    }
    const double q_ready[5] = {0.0, -1.0, 0.5, 0.0, 0.8};
    for (int j = 0; j < 5; ++j) { req.q0[j] = q_ready[j]; req.dq0[j] = 0.0; }
    // 首拍播种:IK@B′ 锚(p_hit 反向 0.8)
    {
      const double yaw = std::atan2(req.p_hit[1] - req.base0[1],
                                    req.p_hit[0] - req.base0[0]);
      const double bx = req.p_hit[0] - 0.8 * std::cos(yaw);
      const double by = req.p_hit[1] - 0.8 * std::sin(yaw);
      const IkResult seed = fx.ik->Solve(req.p_hit, bx, by, 0.0, q_ready);
      if (seed.err < 0.03) {
        for (int j = 0; j < 5; ++j) req.q_seed[j] = seed.q[j];
        req.q_seed_ok = true;
      }
    }
    WmPlan plan;
    REQUIRE_MESSAGE(mpc.Solve(req, &plan), sc.name);
    const double dref = std::hypot(plan.term_base[0] - sc.ref_base[0],
                                   plan.term_base[1] - sc.ref_base[1]);
    INFO(sc.name, " term=(", plan.term_base[0], ",", plan.term_base[1],
         ") ref=", sc.ref_base[0], ",", sc.ref_base[1],
         " dref=", dref, " pos_res=", plan.pos_res, " vel_res=", plan.vel_res,
         " ms=", plan.ms, " iters=", plan.sqp_iters);
    // ① 非线性拍心残差收敛(联合最优:流形可达成,<2cm)
    CHECK(plan.pos_res < 0.02);
    // ② 终端底盘与 python 参考同邻域(±0.15m —— 分支/数值差异容忍)
    CHECK(dref < 0.15);
    // ③ 轨迹合法
    REQUIRE((int)plan.rx.size() == WholebodyMpc::kN + 1);
    for (int k = 0; k <= WholebodyMpc::kN; ++k) {
      CHECK(std::fabs(plan.rvx[k]) <= 1.5 + 1e-6);
      CHECK(std::fabs(plan.rvy[k]) <= 1.5 + 1e-6);
    }
    CHECK(std::fabs(plan.rx[0] - req.base0[0]) < 1e-4);  // OSQP eps 1e-5 口径
  }
}

TEST_CASE("wholebody_mpc: 末段窗守门 + 热启动重解稳定") {
  Fixture fx;
  WholebodyMpc mpc(*fx.ik, fx.m, WmLimits());
  WmRequest req;
  req.t_now = 1.00; req.t_hit = 1.85;
  req.p_hit[0] = 0.95; req.p_hit[1] = -0.55; req.p_hit[2] = 0.70;
  const double v_in[3] = {-5.5, -0.3, -1.7};
  const double el = 15.0 * M_PI / 180.0;
  const double v_out[3] = {8.0 * std::cos(el), 0.0, 8.0 * std::sin(el)};
  double n_star[3];
  ReboundInverse(v_in, v_out, req.v_star, n_star);
  req.base0[0] = -0.05; req.base0[1] = -0.10;
  req.vbase0[0] = 0.8; req.vbase0[1] = -0.4;
  const double q_ready[5] = {0.0, -1.0, 0.5, 0.0, 0.8};
  for (int j = 0; j < 5; ++j) { req.q0[j] = q_ready[j]; req.dq0[j] = 0.0; }
  {
    const double yaw = std::atan2(req.p_hit[1] - req.base0[1],
                                  req.p_hit[0] - req.base0[0]);
    const IkResult seed = fx.ik->Solve(req.p_hit, req.p_hit[0] - 0.8 * std::cos(yaw),
                                       req.p_hit[1] - 0.8 * std::sin(yaw), 0.0, q_ready);
    if (seed.err < 0.03) {
      for (int j = 0; j < 5; ++j) req.q_seed[j] = seed.q[j];
      req.q_seed_ok = true;
    }
  }
  WmPlan p1, p2;
  REQUIRE(mpc.Solve(req, &p1));
  req.t_now = 1.02;
  req.base0[0] += 0.02;   // 底盘微动(20ms)
  REQUIRE(mpc.Solve(req, &p2));
  const double jump = std::hypot(p2.term_base[0] - p1.term_base[0],
                                 p2.term_base[1] - p1.term_base[1]);
  INFO("warm jump=", jump, " p2 ms=", p2.ms);
  CHECK(jump < 0.15);
  // 末段窗
  req.t_now = 1.80;
  WmPlan p3;
  CHECK_FALSE(mpc.Solve(req, &p3));
}
