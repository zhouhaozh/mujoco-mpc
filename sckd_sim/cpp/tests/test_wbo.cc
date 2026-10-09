// 方案 B v2 在线底盘 OCP 协调器单元测试(2026-10-08)。
//   ① 求解成功、毫秒级、解时窗内;
//   ② 可达工况:终端底盘落 anchor*(离线 back*≈0.45)附近;
//   ③ 不可达工况(时窗不足):终端沿可达集滑动而非命令不可达目标
//      (back(z) 蒸馏失败根因的正面断言);
//   ④ 终端速度被 v*_xy 拉动(底盘供能方向 = 拍速需求,非 d̂ 投影);
//   ⑤ 参考轨迹动态可行(每轴速度界内)且起于实测状态;
//   ⑥ 热启动重解稳定 + 末段窗守门。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>
#include <vector>

#include "control/wholebody_ocp.h"

using namespace sckd;

namespace {

// wholebody_ocp.py rebound_inverse 同式(a=1+k=1.5,b=−k=−0.5)
void ReboundInverse(const double v_in[3], const double v_out[3],
                    double v_star[3], double n_star[3]) {
  const double vo = std::sqrt(v_out[0] * v_out[0] + v_out[1] * v_out[1] +
                              v_out[2] * v_out[2]);
  for (int i = 0; i < 3; ++i) n_star[i] = v_out[i] / std::max(vo, 0.01);
  const double v_in_n = v_in[0] * n_star[0] + v_in[1] * n_star[1] + v_in[2] * n_star[2];
  const double vr_n = (vo - (-0.5) * v_in_n) / 1.5;
  for (int i = 0; i < 3; ++i) v_star[i] = n_star[i] * vr_n;
}

WboRequest G1Request() {
  // wholebody_ocp.py V2_SCENARIOS['G1 gs 中央'] 同参数
  WboRequest req;
  req.t_now = 1.00; req.t_hit = 1.85;
  req.p_hit[0] = 0.95; req.p_hit[1] = -0.55; req.p_hit[2] = 0.70;
  const double v_in[3] = {-5.5, -0.3, -1.7};
  const double el = 15.0 * M_PI / 180.0;
  const double v_out[3] = {8.0 * std::cos(el), 0.0, 8.0 * std::sin(el)};
  double n_star[3];
  ReboundInverse(v_in, v_out, req.v_star, n_star);
  req.base0[0] = -0.05; req.base0[1] = -0.10;
  req.vbase0[0] = 0.8; req.vbase0[1] = -0.4;
  return req;
}

}  // namespace

TEST_CASE("wholebody_ocp: G1 可达工况 anchor* 达成 + 轨迹可行") {
  WholebodyOcp wbo;
  WboRequest req = G1Request();
  WboResult r;
  REQUIRE(wbo.Solve(req, &r));
  CHECK(r.ms < 10.0);
  // ② anchor* = p*_xy − 0.45·d̂(离线联合最优后撤;B′ 为 0.8)
  const double dx = req.p_hit[0] - req.base0[0], dy = req.p_hit[1] - req.base0[1];
  const double D = std::hypot(dx, dy);
  const double ax = req.p_hit[0] - 0.45 * dx / D, ay = req.p_hit[1] - 0.45 * dy / D;
  const double da = std::hypot(r.term_base[0] - ax, r.term_base[1] - ay);
  INFO("anchor dist=", da, " ms=", r.ms);
  CHECK(da < 0.10);   // 可达:能量代价换不来大偏移,应贴近锚
  // ④ 终端速度沿 v* 方向有正分量(底盘供能;v*_xy=(3.3,0))
  const double vdot = r.term_vbase[0] * req.v_star[0] + r.term_vbase[1] * req.v_star[1];
  CHECK(vdot > 0.5);
  // ⑤ 轨迹:起于实测、每轴速度界内、末端=终端
  REQUIRE((int)r.rx.size() == WholebodyOcp::kN + 1);
  CHECK(std::fabs(r.rx[0] - req.base0[0]) < 1e-6);
  CHECK(std::fabs(r.rvx[0] - req.vbase0[0]) < 1e-6);
  for (int k = 0; k <= WholebodyOcp::kN; ++k) {
    CHECK(std::fabs(r.rvx[k]) <= 1.5 + 1e-6);
    CHECK(std::fabs(r.rvy[k]) <= 1.5 + 1e-6);
  }
  CHECK(std::fabs(r.rx[WholebodyOcp::kN] - r.term_base[0]) < 1e-9);

  // ⑥ 热启动重解(t 前移 20ms):稳定
  req.t_now = 1.02;
  WboResult r2;
  REQUIRE(wbo.Solve(req, &r2));
  const double jump = std::hypot(r2.term_base[0] - r.term_base[0],
                                 r2.term_base[1] - r.term_base[1]);
  INFO("warm re-solve jump=", jump, " ms=", r2.ms);
  CHECK(jump < 0.05);
}

TEST_CASE("wholebody_ocp: 不可达工况沿可达集滑动(back(z) 失败根因)") {
  WholebodyOcp wbo;
  WboRequest req = G1Request();
  // 时窗砍到 0.35s:anchor*(距起点 ~1.05m)需要均速 3 m/s > 每轴钳 —— 不可达。
  // back(z)=0.45 静态规则的闭环失败正在此(命令不可达 → init-fail/底盘误差
  // 0.93-1.57m);OCP 须沿可达集给最近的可达终端
  req.t_hit = req.t_now + 0.35;
  WboResult r;
  REQUIRE(wbo.Solve(req, &r));
  const double da = std::hypot(r.term_base[0] - req.base0[0], r.term_base[1] - req.base0[1]);
  INFO("unreachable: moved=", da, " |v_N|=", std::hypot(r.term_vbase[0], r.term_vbase[1]));
  // 可达上界 ~ v_axis·T(1.5·0.35=0.525)—— 不得超(双积分器物理一致)
  CHECK(da < 0.525 + 0.05);
  CHECK(da > 0.05);   // 且确实在朝锚努力,而非原地不动
}

TEST_CASE("wholebody_ocp: 前进序列稳定性(底盘逐拍向锚推进,终端不得倒退)") {
  WholebodyOcp wbo;
  WboRequest req = G1Request();
  WboResult r;
  double last_x = -1e9;
  for (int step = 0; step < 6; ++step) {
    req.t_now = 1.00 + 0.1 * step;
    if (step > 0) {   // 底盘朝锚推进 ~1 m/s(世界系)
      req.base0[0] += 0.10;
      req.base0[1] += -0.05;
      req.vbase0[0] = 1.0; req.vbase0[1] = -0.5;
    }
    REQUIRE(wbo.Solve(req, &r));
    MESSAGE("step ", step, " base=(", req.base0[0], ",", req.base0[1],
            ") term=(", r.term_base[0], ",", r.term_base[1],
            ") vN=(", r.term_vbase[0], ",", r.term_vbase[1], ")");
    CHECK(r.term_base[0] > last_x - 0.10);   // 终端不倒退(慢推进=能量剖面常态,
    last_x = r.term_base[0];                 // 闭环爬行病由绝对时刻参考治理)
  }
}

TEST_CASE("wholebody_ocp: 末段窗守门(t_hit−t<0.15 拒收)") {
  WholebodyOcp wbo;
  WboRequest req = G1Request();
  req.t_hit = req.t_now + 0.10;
  WboResult r;
  CHECK_FALSE(wbo.Solve(req, &r));
}
