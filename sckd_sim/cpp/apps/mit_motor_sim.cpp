// MIT 协议单机仿真(2026-10-18):独立复现真机臂执行链的单关节仿真,
// 仅 C++ 标准库,不链 sckd_core/MuJoCo(CMake 单独目标)。
// 链路:上位机 250Hz 发 4×16bit MIT 帧(量化常量=旧仓 robot_control.h
// RS04:P±12.57rad/V±36rad·s⁻¹/kp[0,5000]/kd[0,100];另设 16bit τ 槽
// ±120N·m 模拟"固件带 T 位"假设——现机 MotionSet 无此槽,
// robot_control.cc:3243)→ 驱动器 1kHz 执行 τ=kp(p*−p)+kd(v*−v)+τ_ff →
// 9ms 力矩命令延迟(motors.json 实测)→ RS04 T-N 包络(motors.json 手册
// 读点,力矩随|dq|衰减)+ 臂侧负载(等效惯量 J/粘滞 b/库仑 τc/重力 G·cosθ)。
// 四个同参对象并行对比:A=纯MIT(真机现状) B=+供应商速度环积分(结构假设:
// τ_i=ki·∫(v*−v)dt 限幅 ti_max,待固件工程师确认) C=+理想τ前馈(T位假设,
// 由参考逆动力学 J·q̈*+b·q̇*+τc·tanh+G·cosθ 随帧下发)
// D=前馈+积分。增益缺省=rob robot_sim.ini [Arm] J2 track(kp50/kd30)。
// 三场景各出一 CSV(1ms 行)供 python/tools/plot_mit_motor.py 绘跟踪曲线:
//   step  软阶跃(五次曲线 T=0.15s)+保持段 → 重力下垂/整定
//   chirp 0.5→4Hz 变频正弦(幅0.2rad,偏置0.6)→ 相位滞后-带宽
//   swing 三段快挥 Δ1.2rad,T=0.30/0.20/0.14s → 挥拍工况峰值误差
// 用法:mit_motor_sim [step|chirp|swing|all] [outdir] [k=v ...]
//   可覆盖:J b tc G kp kd ki ti_max delay_ms frame_ms(标称值为占位,
//   真机标定后替换;J/b/τc/G 即待辨识量)。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <tuple>
#include <vector>

namespace {

// RS04 T-N 包络(motors.json motoring_points,72V 峰值口径)
constexpr double kTnX[] = {0.0,    17.099, 19.8,  23.401, 27.0, 28.8,
                           30.599, 32.401, 33.661, 34.2,  36.0};
constexpr double kTnY[] = {120.0, 120.0, 110.0, 95.0, 78.0, 60.0,
                           52.0,  44.0,  32.0,  10.0, 0.0};
constexpr int kTnN = 11;

double clampd(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// 4×16bit 帧量化(与旧仓 MotionSet 一致:浮点→uint16→浮点往返)
double quant16(double x, double lo, double hi) {
  x = clampd(x, lo, hi);
  const double u = std::lround((x - lo) / (hi - lo) * 65535.0);
  return lo + (hi - lo) * u / 65535.0;
}

// |dq| 处可用力矩(T-N 线性插值;超出末端=0,即空载转速自然限速)
double tn_avail(double w) {
  if (w <= 0) return kTnY[0];
  for (int i = 1; i < kTnN; ++i)
    if (w <= kTnX[i]) {
      const double a = (w - kTnX[i - 1]) / (kTnX[i] - kTnX[i - 1]);
      return kTnY[i - 1] + a * (kTnY[i] - kTnY[i - 1]);
    }
  return 0.0;
}

struct Cfg {
  double J = 0.20;       // 等效惯量 kg·m²(J2 量级,占位待辨识)
  double b = 0.03;       // 粘滞 N·m·s/rad
  double tc = 0.5;       // 库仑摩擦 N·m
  double G = 8.0;        // 重力幅值 N·m(τ_g=G·cosθ,占位)
  double kp = 50.0;      // rob robot_sim.ini J2 kp_track
  double kd = 30.0;      // rob robot_sim.ini J2 kd_track
  double ki = 15.0;      // 供应商速度环积分增益(结构假设)
  double ti_max = 12.0;  // 积分项力矩限幅 N·m
  int delay_ms = 9;      // 命令延迟(motors.json)
  int frame_ms = 4;      // 发帧周期 4ms=250Hz
};

struct Plant {
  double q = 0, dq = 0;
};

struct Driver {
  struct Frame { double p = 0, v = 0, kp = 0, kd = 0, tff = 0; } fr;
  std::deque<double> lag;  // 9ms 延迟线(1kHz 入队)
  double integ = 0;        // 供应商速度环积分器 ∫(v*−v)dt

  // 1kHz tick:按最新量化帧执行 MIT 律,返回经延迟线出端的力矩
  double tick(const Plant& p, const Cfg& c, bool use_vi, double* tau_i) {
    const double ep = fr.p - p.q, ev = fr.v - p.dq;
    double tau = fr.kp * ep + fr.kd * ev + fr.tff;
    double ti = 0;
    if (use_vi) {
      integ = clampd(integ + ev * 0.001, -c.ti_max / c.ki, c.ti_max / c.ki);
      ti = c.ki * integ;
      tau += ti;
    }
    if (tau_i) *tau_i = ti;
    lag.push_back(tau);
    if (static_cast<int>(lag.size()) > c.delay_ms) lag.pop_front();
    return lag.front();
  }
};

// 对象一步(0.1ms 子步):T-N 限幅 + 摩擦(平滑库仑+粘滞) + 重力
void step_plant(Plant& p, double tau, const Cfg& c) {
  const double ta = tn_avail(std::fabs(p.dq));
  tau = clampd(tau, -ta, ta);
  const double fr = c.tc * std::tanh(p.dq / 0.05) + c.b * p.dq;
  const double tg = c.G * std::cos(p.q);
  p.dq += (tau - fr - tg) / c.J * 1e-4;
  if (std::fabs(p.dq) < 2e-4 && std::fabs(tau - tg) < c.tc) p.dq = 0;  // 静摩擦锁
  p.q += p.dq * 1e-4;
}

using Ref = std::tuple<double, double, double>;  // (q, dq, ddq)

Ref hold(double q) { return {q, 0, 0}; }

// 五次曲线点到位(s∈[0,1]:10s³−15s⁴+6s⁵),最大加速度=5.7735Δ/T²
Ref quintic(double t, double t0, double T, double q0, double q1) {
  if (t <= t0) return hold(q0);
  if (t >= t0 + T) return hold(q1);
  const double s = (t - t0) / T, d = q1 - q0;
  const double q = q0 + d * (10 * s * s * s - 15 * s * s * s * s +
                             6 * s * s * s * s * s);
  const double v = d * (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
  const double a = d * (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
  return {q, v, a};
}

double seg_q(double t, double t0, double T, double q0, double q1) {
  if (t < t0) return q0;
  if (t >= t0 + T) return q1;
  const double s = (t - t0) / T;
  return q0 + (q1 - q0) * (10 * s * s * s - 15 * s * s * s * s +
                           6 * s * s * s * s * s);
}

Ref RefAt(const std::string& sc, double t) {
  if (sc == "step") {  // 保持→软阶跃族(1.0s −1.0rad/3.0s +1.0/5.0s −0.5)
    if (t < 1.0) return hold(0.6);
    if (t < 3.0) return quintic(t, 1.0, 0.15, 0.6, -0.4);
    if (t < 5.0) return quintic(t, 3.0, 0.15, -0.4, 0.6);
    return quintic(t, 5.0, 0.15, 0.6, 0.1);
  }
  if (sc == "chirp") {  // 8s 线性变频 0.5→4Hz,幅 0.2rad,偏置 0.6
    constexpr double T = 8.0, f0 = 0.5, f1 = 4.0, A = 0.2, q0 = 0.6;
    const double f = f0 + (f1 - f0) * t / T;
    const double w = 2 * M_PI * f, wd = 2 * M_PI * (f1 - f0) / T;
    const double ph = 2 * M_PI * (f0 * t + (f1 - f0) / (2 * T) * t * t);
    return {q0 + A * std::sin(ph), A * w * std::cos(ph),
            A * (wd * std::cos(ph) - w * w * std::sin(ph))};
  }
  // swing:0.8s 起,三段快挥 0.6→−0.6(Δ1.2rad),T=0.30/0.20/0.14s
  if (t < 0.8) return hold(0.6);
  if (t < 2.2) return quintic(t, 0.8, 0.30, 0.6, -0.6);
  if (t < 3.4) return quintic(t, 2.2, 0.20, seg_q(t, 2.2, 0.0, -0.6, -0.6), 0.6);
  return quintic(t, 3.4, 0.14, 0.6, -0.6);
}

double scen_dur(const std::string& sc) {
  if (sc == "step") return 6.0;
  if (sc == "chirp") return 8.0;
  return 5.0;
}

struct VarSpec {
  const char* tag;
  bool vi, ff;
};

constexpr VarSpec kVar[] = {
    {"a", false, false},  // A 纯 MIT(真机现状)
    {"b", true, false},   // B +速度环积分(供应商固件假设)
    {"c", false, true},   // C +理想 τ 前馈(T 位假设)
    {"d", true, true},    // D 前馈+积分
};
constexpr int kNV = 4;

void run_scenario(const std::string& sc, const std::string& outdir,
                  const Cfg& c) {
  std::vector<Plant> pl(kNV);
  std::vector<Driver> dr(kNV);
  {  // 对象初始化在参考起点(消除起始瞬态污染)
    const auto [q0, v0, a0] = RefAt(sc, 0);
    for (auto& p : pl) { p.q = q0; p.dq = v0; }
    (void)a0;
  }

  const std::string path = outdir + "/mit_" + sc + ".csv";
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) { std::printf("[FAIL] open %s\n", path.c_str()); return; }
  std::fprintf(f, "t,q_ref,dq_ref,ddq_ref,tau_ff_ideal");
  for (const auto& v : kVar)
    std::fprintf(f, ",q_%s,dq_%s,err_%s,tau_%s", v.tag, v.tag, v.tag, v.tag);
  std::fprintf(f, ",taui_b,taui_d\n");

  const int n_ms = static_cast<int>(scen_dur(sc) * 1000);
  double taui[kNV] = {0, 0, 0, 0}, tau_now[kNV] = {0, 0, 0, 0};
  for (int ms = 1; ms <= n_ms && f; ++ms) {
    const double t = ms * 1e-3;
    const auto [qr, dqr, ddqr] = RefAt(sc, t);
    // 理想 τ 前馈(参考逆动力学;C/D 随帧量化下发,A/B 无槽不发)
    const double tff = c.J * ddqr + c.b * dqr + c.tc * std::tanh(dqr / 0.05) +
                       c.G * std::cos(qr);
    if ((ms - 1) % c.frame_ms == 0) {  // 250Hz 发帧(量化)
      for (int k = 0; k < kNV; ++k) {
        dr[k].fr.p = quant16(qr, -12.57, 12.57);
        dr[k].fr.v = quant16(dqr, -36.0, 36.0);
        dr[k].fr.kp = quant16(c.kp, 0, 5000);
        dr[k].fr.kd = quant16(c.kd, 0, 100);
        dr[k].fr.tff = kVar[k].ff ? quant16(tff, -120, 120) : 0.0;
      }
    }
    for (int k = 0; k < kNV; ++k) {  // 1kHz 驱动器 + 10×0.1ms 物理
      tau_now[k] = dr[k].tick(pl[k], c, kVar[k].vi, &taui[k]);
      for (int s = 0; s < 10; ++s) step_plant(pl[k], tau_now[k], c);
    }
    std::fprintf(f, "%.4f,%.5f,%.4f,%.3f,%.2f", t, qr, dqr, ddqr, tff);
    for (int k = 0; k < kNV; ++k)
      std::fprintf(f, ",%.5f,%.4f,%.5f,%.2f", pl[k].q, pl[k].dq,
                   pl[k].q - qr, tau_now[k]);
    std::fprintf(f, ",%.2f,%.2f\n", taui[1], taui[3]);
  }
  std::fclose(f);
  std::printf("[OK] %-5s %d rows -> %s\n", sc.c_str(), n_ms, path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string sc = argc > 1 ? argv[1] : "all";
  const std::string outdir = argc > 2 ? argv[2] : "eval/reports";
  Cfg c;
  for (int i = 3; i < argc; ++i) {  // k=v 覆盖(J/b/tc/G/kp/kd/ki/ti_max/...)
    const std::string s = argv[i];
    const auto eq = s.find('=');
    if (eq == std::string::npos) continue;
    const double v = std::atof(s.c_str() + eq + 1);
    const std::string k = s.substr(0, eq);
    if (k == "J") c.J = v; else if (k == "b") c.b = v;
    else if (k == "tc") c.tc = v; else if (k == "G") c.G = v;
    else if (k == "kp") c.kp = v; else if (k == "kd") c.kd = v;
    else if (k == "ki") c.ki = v; else if (k == "ti_max") c.ti_max = v;
    else if (k == "delay_ms") c.delay_ms = static_cast<int>(v);
    else if (k == "frame_ms") c.frame_ms = static_cast<int>(v);
  }
  std::printf("[cfg] J=%.3f b=%.3f tc=%.2f G=%.1f kp=%.0f kd=%.0f ki=%.1f "
              "ti_max=%.1f delay=%dms frame=%dms\n",
              c.J, c.b, c.tc, c.G, c.kp, c.kd, c.ki, c.ti_max, c.delay_ms,
              c.frame_ms);
  if (sc == "all") {
    for (const char* s : {"step", "chirp", "swing"}) run_scenario(s, outdir, c);
  } else {
    run_scenario(sc, outdir, c);
  }
  return 0;
}
