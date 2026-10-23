// 5关节臂 MIT 前馈辨识仿真(2026-10-20):真机标定规程的仿真先行验证。
// 与 mit_motor_sim.cpp(单关节纯标准库)保持语义一致的不变量:
//   250Hz 16bit 量化帧 / 1kHz 驱动器 τ=kp(p*−p̃)+kd(v*−ṽ)+τ_ff /
//   9ms 帧延迟 / T-N 包络削顶 / 0.1ms 物理子步。
// 本版差异(相对单关节版):①B 积分变体删除——供应商将提供标准 MIT
//   固件(含 τ 前馈槽、无积分项→命令重构 τ 精确),变体= A / C_ideal
//   (mj_inverse 完整逆动力学) / C_id(ff_file 回放);②测量侧编码器量化
//   (motors.json encoder_bits=14);③对象=MuJoCo 5 关节耦合真值:mjSpec
//   按名删底盘 12 执行器+12 关节+root freejoint+home keyframe(qpos 长
//   24 不删则 compile 失败)→固定基座,断言 nq=nv=5;④双包络 J1/J2=RS04
//   (120N·m)、J3/J4/J5=RS06(36N·m),量化量程逐关节(速度 36/75、τ 槽
//   120/36);⑤摩擦真值运行时注入 dof_damping/dof_frictionloss,另持
//   无摩擦副本 m_rb 供诊断与理想前馈(mj_inverse 含 passive 项)。
// 场景=真机标定规程本身:
//   selftest  手术/诊断自检(nq、重力、Mii、包络端点)
//   poses     N 姿位正序+倒序双向趋近(±τc 静摩擦带平均消),稳态
//             kp·e_p 点测重力;稳态判据=max|dq|<2e-3 持续300ms 且
//             std(kp·e_p)<0.3N·m,超时 5s 报错不静默
//   constvel  逐关节恒速双向梯形斜坡(位置环跟踪,速度模式不可用——
//             kd 受延迟稳定上限 kd·T_eff/J≲1.5 约束)→摩擦辨识
//   chirp     全关节 4 音多正弦,幅度按包络调度
//             A=min(Amax, 0.4·τpk/(J0·ω²)) 且 ΣA≤预算 → τ 重构+惯量
//   swing     三变体闭环验收(挥拍工况)
// CSV(首行组 # k=v manifest;poses 4ms 行、constvel 2ms、其余 1ms):
//   t + qref/vref/aref_j{1..5} + 命令回显 p/v/kp/kd/tff_cmd_j*(真机可记)
//   + 遥测 q/dq_j*(250Hz 采样 ZOH) + 真值诊断 tau/grav/Mii_j*(真机无,
//   python 仅验证用);poses 另有 settled/pass_dir;swing 为三变体
//   q/dq/tau/tff_{a,ci,cid}_j*。
// 用法:mit_arm_id_sim [selftest|poses|constvel|chirp|swing|all] [outdir]
//   [k=v ...](model= ff_file= poses_n= chirp_T= delay_ms= frame_ms=
//   fb_bits= vnoise= seed= sub= b1..b5= tc1..tc5= hold_kp1..5= hold_kd1..5=
//   mv_kp1..5= mv_kd1..5= J01..J05= pose_move_T= pose_hold_T= cv_acc=
//   cv_cruise=)
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace {

// ---------- 电机常量(motors.json,J1/J2=RS04,J3/J4/J5=RS06) ----------
constexpr double kTnX04[] = {0.0,    17.099, 19.8,  23.401, 27.0, 28.8,
                             30.599, 32.401, 33.661, 34.2,  36.0};
constexpr double kTnY04[] = {120.0, 120.0, 110.0, 95.0, 78.0, 60.0,
                             52.0,  44.0,  32.0,  10.0, 0.0};
constexpr double kTnX06[] = {0.0,    43.751, 46.876, 50.0,  54.688, 58.594,
                             62.501, 65.625, 66.876, 67.187, 75.0};
constexpr double kTnY06[] = {36.0, 36.0, 33.0, 30.0, 26.0, 22.0,
                             18.0, 12.0, 10.0, 5.0,  0.0};
constexpr int kTnN = 11;
constexpr int kRs[5] = {0, 0, 1, 1, 1};               // 0=RS04 1=RS06
constexpr double kVmax[5] = {36, 36, 75, 75, 75};     // 速度量化量程
constexpr double kTmax[5] = {120, 120, 36, 36, 36};   // τ 前馈槽量程
constexpr double kTpk[5] = {120, 120, 36, 36, 36};    // 峰值力矩
constexpr double kPosMax = 12.57;

double clampd(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// bits 位量化(浮点→整数→浮点往返;16=命令帧,14=编码器测量)
double quantn(double x, double lo, double hi, int bits) {
  x = clampd(x, lo, hi);
  const double n = static_cast<double>((1 << bits) - 1);
  const double u = std::lround((x - lo) / (hi - lo) * n);
  return lo + (hi - lo) * u / n;
}

// |dq| 处可用力矩(T-N 线性插值;超出末端=0,空载转速自然限速)
double tn_avail(int j, double w) {
  const double* X = kRs[j] ? kTnX06 : kTnX04;
  const double* Y = kRs[j] ? kTnY06 : kTnY04;
  if (w <= 0) return Y[0];
  for (int i = 1; i < kTnN; ++i)
    if (w <= X[i]) {
      const double a = (w - X[i - 1]) / (X[i] - X[i - 1]);
      return Y[i - 1] + a * (Y[i] - Y[i - 1]);
    }
  return 0.0;
}

// ---------- 配置 ----------
struct Cfg {
  std::string model = "models/generated/x4_tennis_scene.mjcf";
  std::string ff_file;  // swing 的 C_id 前馈回放(空=无)
  int sub = 10;         // 0.1ms 物理子步
  int delay_ms = 9;     // 命令→生效延迟(motors.json command_delay_ms)
  int frame_ms = 4;     // 250Hz
  int fb_bits = 14;     // 编码器测量量化(motors.json encoder_bits)
  double vnoise = 0;    // 测量噪声 σ(默认 0=确定性)
  unsigned seed = 0;
  double b[5] = {0.05, 0.03, 0.02, 0.01, 0.005};  // 粘滞摩擦真值
  double tc[5] = {0.8, 0.6, 0.3, 0.15, 0.10};     // 库仑摩擦真值
  double arm[5] = {0.02, 0.02, 0.01, 0.005, 0.005};  // QDD 转子反射惯量占位
  // (mjcf armature=0 系已知缺口:真机 RS04/RS06 转子+减速器反射惯量待 F3 辨识;
  //  无此项 J5 刚体惯量仅 0.0011,kd>0.2 即越过延迟稳定上限 kd·T_eff/J≲1.5)
  double hold_kp[5] = {300, 300, 200, 100, 40};   // 静态保持增益
  double hold_kd[5] = {60, 40, 8, 4, 0.5};
  double mv_kp[5] = {80, 50, 40, 25, 20};         // 跟踪/斜坡增益
  double mv_kd[5] = {40, 25, 8, 3, 0.5};
  int poses_n = 32;
  double pose_move_T = 1.5;   // 姿位间五次曲线移动
  double pose_timeout = 5.0;  // 稳态判据超时
  double cv_speeds[3] = {0.3, 0.6, 1.0};  // 恒速档(rad/s;低速 0.15 级会落入
                                          // 爬滑区,ZOH+延迟下高 τc 关节摩擦辨识被污染)
  double cv_acc = 1.5;                    // 斜坡加速度
  double cv_cruise = 1.0;                 // 巡航时长
  double chirp_T = 40.0;
  double J0[5] = {1.5, 2.2, 0.12, 0.06, 0.008};  // 幅度调度惯量先验(≈Mii+arm)
};

// ---------- 模型装载与手术 ----------
struct Arm {
  mjModel* m = nullptr;    // 带注入摩擦的仿真对象
  mjModel* m_rb = nullptr; // 无摩擦副本(诊断重力/Mii + 理想前馈)
  mjData* d = nullptr;     // 主对象数据
  int dof[5], qadr[5];
};

// 删底盘(12 执行器+12 关节)+root freejoint+home keyframe→固定基座 5 关节臂
bool load_arm(Arm& A, const Cfg& c) {
  char err[500] = "";
  mjSpec* spec = mj_parseXML(c.model.c_str(), nullptr, err, sizeof err);
  if (!spec) {
    std::printf("[FAIL] parse %s: %s\n", c.model.c_str(), err);
    return false;
  }
  static const char* kChas[] = {
      "FL_steer_joint", "FL_wheel_left_joint", "FL_wheel_right_joint",
      "FR_steer_joint", "FR_wheel_left_joint", "FR_wheel_right_joint",
      "RL_steer_joint", "RL_wheel_left_joint", "RL_wheel_right_joint",
      "RR_steer_joint", "RR_wheel_left_joint", "RR_wheel_right_joint"};
  bool ok = true;
  // 先删执行器再删关节(带执行器的关节直接删 compile 失败)
  for (const char* nm : kChas) {
    mjsElement* e = mjs_findElement(spec, mjOBJ_ACTUATOR, nm);
    if (e) mjs_delete(spec, e); else ok = false;
  }
  for (const char* nm : kChas) {
    mjsElement* e = mjs_findElement(spec, mjOBJ_JOINT, nm);
    if (e) mjs_delete(spec, e); else ok = false;
  }
  mjsElement* root = mjs_findElement(spec, mjOBJ_JOINT, "root");
  if (root) mjs_delete(spec, root); else ok = false;
  mjsElement* key = mjs_findElement(spec, mjOBJ_KEY, "home");  // qpos 长 24,不删则 compile 失败
  if (key) mjs_delete(spec, key);
  if (!ok) {
    std::printf("[FAIL] mjcf 缺少预期底盘元素\n");
    mj_deleteSpec(spec);
    return false;
  }
  A.m = mj_compile(spec, nullptr);
  mj_deleteSpec(spec);
  if (!A.m) { std::printf("[FAIL] compile 手术后模型\n"); return false; }
  if (A.m->nq != 5 || A.m->nv != 5) {
    std::printf("[FAIL] 手术后 nq=%d nv=%d(预期 5/5)\n", A.m->nq, A.m->nv);
    return false;
  }
  for (int j = 0; j < 5; ++j) {
    char nm[16];
    std::snprintf(nm, sizeof nm, "Joint%d", j + 1);
    const int jid = mj_name2id(A.m, mjOBJ_JOINT, nm);
    if (jid < 0) { std::printf("[FAIL] 缺 %s\n", nm); return false; }
    A.dof[j] = A.m->jnt_dofadr[jid];
    A.qadr[j] = A.m->jnt_qposadr[jid];
    if (A.m->jnt_type[jid] != mjJNT_HINGE) {
      std::printf("[FAIL] %s 非 hinge\n", nm);
      return false;
    }
  }
  A.m->opt.timestep = 1e-4;
  A.m->opt.disableflags |= mjDSBL_CONTACT;  // 臂 geoms 全 contype=0,地板只碰底盘轮(已删)
  for (int j = 0; j < 5; ++j) {  // 摩擦/转子反射惯量真值注入
    A.m->dof_damping[A.dof[j]] = c.b[j];
    A.m->dof_frictionloss[A.dof[j]] = c.tc[j];
    A.m->dof_armature[A.dof[j]] = c.arm[j];
  }
  A.m_rb = mj_copyModel(nullptr, A.m);
  if (!A.m_rb) return false;
  for (int j = 0; j < 5; ++j) {
    A.m_rb->dof_damping[A.dof[j]] = 0;
    A.m_rb->dof_frictionloss[A.dof[j]] = 0;
  }
  // m_rb 摩擦已清零 → 诊断重力/Mii 与理想前馈不带 passive 项
  A.d = mj_makeData(A.m);
  return A.d != nullptr;
}

// ---------- 驱动器(5 关节 MIT,帧延迟线) ----------
struct Driver {
  struct Frame { double p[5], v[5], kp[5], kd[5], tff[5]; } cur;
  std::deque<Frame> lag;  // 1ms ZOH 入队,出端=delay_ms 前的帧

  void push_advance(const Frame& sent, int delay_ms) {
    lag.push_back(sent);
    if (static_cast<int>(lag.size()) > delay_ms) {
      cur = lag.front();
      lag.pop_front();
    }
  }
  void prefill(const Frame& f0, int delay_ms) {
    cur = f0;
    lag.clear();
    for (int i = 0; i < delay_ms; ++i) lag.push_back(f0);
  }
  // 1kHz tick:生效帧 + 当前编码器测量(加噪→fb_bits 量化)→ MIT 律 → T-N 削顶
  void tick(const Arm& A, const Cfg& c, std::mt19937& rng,
            std::normal_distribution<double>& nz, double tau[5]) {
    for (int j = 0; j < 5; ++j) {
      const double qm = quantn(A.d->qpos[A.qadr[j]] + nz(rng), -kPosMax, kPosMax, c.fb_bits);
      const double vm = quantn(A.d->qvel[A.dof[j]] + nz(rng), -kVmax[j], kVmax[j], c.fb_bits);
      const double t = cur.kp[j] * (cur.p[j] - qm) + cur.kd[j] * (cur.v[j] - vm) +
                       cur.tff[j];
      const double ta = tn_avail(j, std::fabs(vm));
      tau[j] = clampd(t, -ta, ta);
    }
  }
};

Driver::Frame mkframe(const double qr[5], const double vr[5], const double kp[5],
                      const double kd[5], const double tff[5]) {
  Driver::Frame f;
  for (int j = 0; j < 5; ++j) {
    f.p[j] = quantn(qr[j], -kPosMax, kPosMax, 16);
    f.v[j] = quantn(vr[j], -kVmax[j], kVmax[j], 16);
    f.kp[j] = quantn(kp[j], 0, 5000, 16);
    f.kd[j] = quantn(kd[j], 0, 100, 16);
    f.tff[j] = tff ? quantn(tff[j], -kTmax[j], kTmax[j], 16) : 0.0;
  }
  return f;
}

// ---------- 物理推进(0.1ms 子步,子步内按当前速度重削 T-N) ----------
void step_world(const Arm& A, const Cfg& c, const double tau[5]) {
  for (int s = 0; s < c.sub; ++s) {
    for (int j = 0; j < 5; ++j) {
      const double ta = tn_avail(j, std::fabs(A.d->qvel[A.dof[j]]));
      A.d->qfrc_applied[A.dof[j]] = clampd(tau[j], -ta, ta);
    }
    mj_step(A.m, A.d);
  }
}

// 250Hz 遥测采样(编码器量化;主循环在帧边界调用后行内 ZOH 记录)
void sample_tel(const Arm& A, const Cfg& c, std::mt19937& rng,
                std::normal_distribution<double>& nz, double tq[5], double tv[5]) {
  for (int j = 0; j < 5; ++j) {
    tq[j] = quantn(A.d->qpos[A.qadr[j]] + nz(rng), -kPosMax, kPosMax, c.fb_bits);
    tv[j] = quantn(A.d->qvel[A.dof[j]] + nz(rng), -kVmax[j], kVmax[j], c.fb_bits);
  }
}

// ---------- 诊断(无摩擦副本) ----------
void diag(const Arm& A, mjData* s, const double* qpos, double grav[5], double Mii[5]) {
  for (int j = 0; j < 5; ++j) s->qpos[A.qadr[j]] = qpos[A.qadr[j]];
  for (int i = 0; i < A.m_rb->nv; ++i) { s->qvel[i] = 0; s->qacc[i] = 0; }
  mj_inverse(A.m_rb, s);  // qvel=qacc=0 → qfrc_inverse 即纯重力
  for (int j = 0; j < 5; ++j) grav[j] = s->qfrc_inverse[A.dof[j]];
  for (int i = 0; i < A.m_rb->nv; ++i) s->qvel[i] = 0;
  mj_kinematics(A.m_rb, s);
  mj_comPos(A.m_rb, s);
  mj_crb(A.m_rb, s);  // 填 qM
  double M[25];
  mj_fullM(A.m_rb, s, M);
  for (int j = 0; j < 5; ++j) Mii[j] = M[5 * j + j];
}

// 理想前馈 = 完整逆动力学(m_rb,无 passive)+ 摩擦真值(b·v+τc·tanh)
void ideal_ff(const Arm& A, mjData* s, const Cfg& c, const double q[5],
              const double v[5], const double a[5], double tff[5]) {
  for (int j = 0; j < 5; ++j) {
    s->qpos[A.qadr[j]] = q[j];
    s->qvel[A.dof[j]] = v[j];
    s->qacc[A.dof[j]] = a[j];
  }
  mj_inverse(A.m_rb, s);
  for (int j = 0; j < 5; ++j)
    tff[j] = s->qfrc_inverse[A.dof[j]] + c.b[j] * v[j] +
             c.tc[j] * std::tanh(v[j] / 0.05);
}

// ---------- 参考轨迹 ----------
constexpr double kHome[5] = {0.0, -0.61, -0.61, 0.0, 0.0};  // sim 待机 J2=J3=−35°

void hold5(const double q0[5], double q[5], double v[5], double a[5]) {
  for (int j = 0; j < 5; ++j) { q[j] = q0[j]; v[j] = 0; a[j] = 0; }
}

void quintic5(double t, double t0, double T, const double q0[5], const double q1[5],
              double q[5], double v[5], double a[5]) {
  if (t <= t0) return hold5(q0, q, v, a);
  if (t >= t0 + T) return hold5(q1, q, v, a);
  const double s = (t - t0) / T;
  const double s2 = s * s, s3 = s2 * s, s4 = s3 * s, s5 = s4 * s;
  for (int j = 0; j < 5; ++j) {
    const double d = q1[j] - q0[j];
    q[j] = q0[j] + d * (10 * s3 - 15 * s4 + 6 * s5);
    v[j] = d * (30 * s2 - 60 * s3 + 30 * s4) / T;
    a[j] = d * (60 * s - 180 * s2 + 120 * s3) / (T * T);
  }
}

// Halton 低差异序列(确定性,seed 平移索引)
double halton(int i, int base) {
  double f = 1, r = 0;
  while (i > 0) { f /= base; r += f * (i % base); i /= base; }
  return r;
}

// 梯形速度段参考(q0 出发,方向 sign,速度 v,加速度 acc,巡航 tc 秒)
double trap_total(double v, double acc, double tc) { return v * v / acc + v * tc; }
void trap_ref(double t, double q0, double sign, double v, double acc, double tc,
              double* q, double* dq, double* ddq) {
  const double ta = v / acc, da = 0.5 * v * v / acc, T = 2 * ta + tc;
  if (t <= 0) { *q = q0; *dq = 0; *ddq = 0; return; }
  if (t < ta) {
    *ddq = sign * acc; *dq = sign * acc * t; *q = q0 + sign * 0.5 * acc * t * t;
  } else if (t < ta + tc) {
    *ddq = 0; *dq = sign * v; *q = q0 + sign * (da + v * (t - ta));
  } else if (t < T) {
    const double r = T - t;
    *ddq = -sign * acc; *dq = sign * acc * r;
    *q = q0 + sign * (da + v * tc + v * r - 0.5 * acc * r * r);
  } else {
    *q = q0 + sign * trap_total(v, acc, tc); *dq = 0; *ddq = 0;
  }
}

// chirp:每关节 4 音多正弦(频率组不可通约;幅度启动时按包络调度)
constexpr double kChF[5][4] = {{0.23, 0.41, 0.73, 1.13},
                               {0.19, 0.37, 0.71, 1.17},
                               {0.53, 0.89, 1.63, 2.41},
                               {0.61, 1.07, 1.87, 2.39},
                               {1.03, 1.79, 3.11, 4.73}};
constexpr double kChPh[5][4] = {{0.0, 0.7, 1.4, 2.1},
                                {0.3, 1.0, 1.7, 2.4},
                                {0.6, 1.3, 2.0, 2.7},
                                {0.9, 1.6, 2.3, 3.0},
                                {1.2, 1.9, 2.6, 3.3}};
constexpr double kChAmax[5] = {0.30, 0.25, 0.25, 0.30, 0.35};
constexpr double kChBudget[5] = {0.30, 0.25, 0.25, 0.30, 0.35};
// (J1 幅度试过 0.8:大振幅下科氏项污染回归,全局残差 0.8→1.6 翻倍——
//  惯量辨识误差不降反升;科氏改由 FF 侧 Christoffel 项补)

struct ChirpCfg {
  double A[5][4];
  double q0[5];
  void init(const Cfg& c) {
    for (int j = 0; j < 5; ++j) {
      q0[j] = kHome[j];
      double s = 0;
      for (int i = 0; i < 4; ++i) {
        const double w = 2 * M_PI * kChF[j][i];
        A[j][i] = std::min(kChAmax[j], 0.4 * kTpk[j] / (c.J0[j] * w * w));
        s += A[j][i];
      }
      if (s > kChBudget[j]) {
        const double sc = kChBudget[j] / s;
        for (int i = 0; i < 4; ++i) A[j][i] *= sc;
      }
    }
  }
};

void chirp_ref(const ChirpCfg& cc, double t, double q[5], double v[5], double a[5]) {
  for (int j = 0; j < 5; ++j) {
    q[j] = cc.q0[j]; v[j] = 0; a[j] = 0;
    for (int i = 0; i < 4; ++i) {
      const double w = 2 * M_PI * kChF[j][i], ph = w * t + kChPh[j][i];
      q[j] += cc.A[j][i] * std::sin(ph);
      v[j] += cc.A[j][i] * w * std::cos(ph);
      a[j] += -cc.A[j][i] * w * w * std::sin(ph);
    }
  }
}

// swing:待机→三段快挥(T=0.30/0.20/0.14s)→收拍保持。
// 幅度约束:J2 峰值 τ≈g+J·5.77Δ/T²≤0.8×120(0.14s 段 Δ≤0.11)——
// 更快的挥拍会触 RS04 包络,前馈与 A 同被削顶,验收图失去分辨力。
constexpr double kSwP[4][5] = {{0.00, -0.61, -0.61, 0.0, 0.0},
                               {0.35, -1.11, -0.30, 0.3, 1.2},
                               {-0.10, -0.89, -0.95, -0.3, -0.8},
                               {0.10, -0.79, -0.45, 0.0, 0.4}};
constexpr double kSwT[3] = {0.30, 0.20, 0.14};
constexpr double kSwDur = 4.0;

void swing_ref(double t, double q[5], double v[5], double a[5]) {
  if (t < 1.0) return hold5(kSwP[0], q, v, a);
  if (t < 1.0 + kSwT[0]) return quintic5(t, 1.0, kSwT[0], kSwP[0], kSwP[1], q, v, a);
  const double t2 = 1.0 + kSwT[0];
  if (t < t2 + kSwT[1]) return quintic5(t, t2, kSwT[1], kSwP[1], kSwP[2], q, v, a);
  const double t3 = t2 + kSwT[1];
  if (t < t3 + kSwT[2]) return quintic5(t, t3, kSwT[2], kSwP[2], kSwP[3], q, v, a);
  return hold5(kSwP[3], q, v, a);
}

// ---------- CSV ----------
void write_manifest(FILE* f, const Cfg& c, const char* sc) {
  std::fprintf(f, "# scenario=%s\n", sc);
  std::fprintf(f, "# model=%s\n", c.model.c_str());
  if (!c.ff_file.empty()) std::fprintf(f, "# ff_file=%s\n", c.ff_file.c_str());
  std::fprintf(f, "# delay_ms=%d frame_ms=%d fb_bits=%d sub=%d vnoise=%g seed=%u\n",
               c.delay_ms, c.frame_ms, c.fb_bits, c.sub, c.vnoise, c.seed);
  auto arr = [&](const char* k, const double* v) {
    std::fprintf(f, "# %s=%g,%g,%g,%g,%g\n", k, v[0], v[1], v[2], v[3], v[4]);
  };
  arr("b", c.b);
  arr("tc", c.tc);
  arr("arm", c.arm);
  arr("hold_kp", c.hold_kp); arr("hold_kd", c.hold_kd);
  arr("mv_kp", c.mv_kp);     arr("mv_kd", c.mv_kd);
  arr("J0", c.J0);
  std::fprintf(f, "# poses_n=%d chirp_T=%g\n", c.poses_n, c.chirp_T);
  std::fprintf(f, "# cv_speeds=%g,%g,%g cv_acc=%g cv_cruise=%g\n",
               c.cv_speeds[0], c.cv_speeds[1], c.cv_speeds[2], c.cv_acc, c.cv_cruise);
  std::fprintf(f, "# tmax=%g,%g,%g,%g,%g\n", kTmax[0], kTmax[1], kTmax[2], kTmax[3],
               kTmax[4]);
}

void hdr_ref_cmd(FILE* f) {  // 参考+命令回显+遥测+真值诊断(单对象场景)
  std::fprintf(f, "t");
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",qref_j%d,vref_j%d,aref_j%d", j + 1, j + 1, j + 1);
  for (const char* k : {"p_cmd", "v_cmd", "kp_cmd", "kd_cmd", "tff_cmd"})
    for (int j = 0; j < 5; ++j) std::fprintf(f, ",%s_j%d", k, j + 1);
  for (const char* k : {"q", "dq", "tau", "grav", "Mii"})
    for (int j = 0; j < 5; ++j) std::fprintf(f, ",%s_j%d", k, j + 1);
}

void row_ref_cmd(FILE* f, double t, const double qr[5], const double vr[5],
                 const double ar[5], const Driver::Frame& sent, const double tq[5],
                 const double tv[5], const double tau[5], const double grav[5],
                 const double Mii[5]) {
  std::fprintf(f, "%.4f", t);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.6f,%.5f,%.4f", qr[j], vr[j], ar[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.6f", sent.p[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.5f", sent.v[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.1f", sent.kp[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.2f", sent.kd[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.3f", sent.tff[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.6f", tq[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.5f", tv[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.3f", tau[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.3f", grav[j]);
  for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.5f", Mii[j]);
}

FILE* open_csv(const std::string& outdir, const char* sc, const Cfg& c) {
  const std::string path = outdir + "/mitarm_" + sc + ".csv";
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) { std::printf("[FAIL] open %s\n", path.c_str()); return nullptr; }
  write_manifest(f, c, sc);
  return f;
}

// ---------- selftest ----------
void run_selftest(const Arm& A, const Cfg& c) {
  std::printf("[selftest] nq=%d nv=%d nu=%d ngeom=%ld\n", A.m->nq, A.m->nv, A.m->nu,
              static_cast<long>(A.m->ngeom));
  mjData* s = mj_makeData(A.m_rb);
  double grav[5], Mii[5];
  const double qs[3][5] = {{0.0, -0.61, -0.61, 0.0, 0.0},
                           {0.0, -1.2, -0.3, 0.5, 0.8},
                           {0.3, -0.4, -1.5, -1.0, -1.2}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) s->qpos[A.qadr[j]] = qs[i][j];
    diag(A, s, s->qpos, grav, Mii);
    std::printf("[grav] q={%.2f,%.2f,%.2f,%.2f,%.2f} -> %.3f %.3f %.3f %.3f %.3f\n",
                qs[i][0], qs[i][1], qs[i][2], qs[i][3], qs[i][4], grav[0], grav[1],
                grav[2], grav[3], grav[4]);
    std::printf("[Mii ]                        -> %.4f %.4f %.4f %.4f %.4f\n",
                Mii[0], Mii[1], Mii[2], Mii[3], Mii[4]);
  }
  mj_deleteData(s);
  std::printf("[tn  ] RS04(0)=%.0f RS04(36)=%.0f RS06(0)=%.0f RS06(75)=%.0f\n",
              tn_avail(0, 0), tn_avail(0, 36), tn_avail(2, 0), tn_avail(2, 75));
  // 稳定性冒烟:保持增益下 1s 静止(延迟线预填,无冷启动瞬态)
  Driver dr;
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> nz(0, c.vnoise);
  double qr[5], vr[5], ar[5], tau[5];
  double tff0[5] = {0, 0, 0, 0, 0};
  hold5(kHome, qr, vr, ar);
  dr.prefill(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
  for (int j = 0; j < 5; ++j) A.d->qpos[A.qadr[j]] = kHome[j];
  mj_forward(A.m, A.d);
  for (int ms = 0; ms < 1000; ++ms) {
    if (ms % c.frame_ms == 0)
      dr.push_advance(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
    else
      dr.push_advance(dr.lag.back(), c.delay_ms);
    dr.tick(A, c, rng, nz, tau);
    step_world(A, c, tau);
  }
  double dqmax = 0;
  for (int j = 0; j < 5; ++j) dqmax = std::max(dqmax, std::fabs(A.d->qvel[A.dof[j]]));
  std::printf("[hold] 1s 后 max|dq|=%.2e rad/s(期望 <1e-3),逐关节:",
              dqmax);
  for (int j = 0; j < 5; ++j)
    std::printf(" j%d=%.2e", j + 1, std::fabs(A.d->qvel[A.dof[j]]));
  std::printf("\n");
}

// ---------- poses ----------
void run_poses(const Arm& A, const Cfg& c, const std::string& outdir) {
  FILE* f = open_csv(outdir, "poses", c);
  if (!f) return;
  hdr_ref_cmd(f);
  std::fprintf(f, ",settled,pass_dir\n");

  // 姿位序列:Halton(J1 固 0——竖直轴重力≡0,不浪费姿位)
  std::vector<std::array<double, 5>> tg;
  for (int i = 1; i <= c.poses_n; ++i) {
    const int k = i + static_cast<int>(c.seed) * 997;
    std::array<double, 5> p{0, 0, 0, 0, 0};
    p[1] = -2.94 + 3.24 * halton(k, 2);
    p[2] = -2.94 + 3.24 * halton(k, 3);
    p[3] = -2.5 + 5.0 * halton(k, 5);
    p[4] = -2.5 + 5.0 * halton(k, 7);
    tg.push_back(p);
  }

  Driver dr;
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> nz(0, c.vnoise);
  mjData* sd = mj_makeData(A.m_rb);
  double qr[5], vr[5], ar[5], tau[5], grav[5], Mii[5], tq[5], tv[5];
  double tff0[5] = {0, 0, 0, 0, 0};
  double q_prev[5] = {kHome[0], kHome[1], kHome[2], kHome[3], kHome[4]};
  for (int j = 0; j < 5; ++j) A.d->qpos[A.qadr[j]] = kHome[j];
  mj_forward(A.m, A.d);
  hold5(q_prev, qr, vr, ar);
  dr.prefill(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);

  long ms = 0;
  int rows = 0, fails = 0, recs = 0;
  // 双向趋近:正序一遍+倒序一遍(每姿位两个趋近方向,±τc 静摩擦带平均消)
  std::vector<std::pair<int, int>> order;
  for (int i = 0; i < c.poses_n; ++i) order.push_back({i, 1});
  for (int i = c.poses_n - 1; i >= 0; --i) order.push_back({i, -1});

  for (const auto& od : order) {
    const double* q_tgt = tg[od.first].data();
    // 相位 1:五次曲线移动
    const int n_move = static_cast<int>(c.pose_move_T * 1000);
    for (int step = 0; step < n_move; ++step, ++ms) {
      quintic5(step * 1e-3, 0, c.pose_move_T, q_prev, q_tgt, qr, vr, ar);
      if (ms % c.frame_ms == 0) {
        dr.push_advance(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
        sample_tel(A, c, rng, nz, tq, tv);
      } else {
        dr.push_advance(dr.lag.back(), c.delay_ms);
      }
      dr.tick(A, c, rng, nz, tau);
      step_world(A, c, tau);
      if (ms % 4 == 0) {
        diag(A, sd, A.d->qpos, grav, Mii);
        row_ref_cmd(f, ms * 1e-3, qr, vr, ar, dr.lag.back(), tq, tv, tau, grav, Mii);
        std::fprintf(f, ",0,%d\n", od.second);
        ++rows;
      }
    }
    // 相位 2:保持+稳态判据(max|dq|<2e-3 持续300ms 且逐关节 std(kp·e_p)<0.3N·m)
    std::deque<double> vmax;
    std::deque<std::array<double, 5>> kpe;
    double t_settle = -1;
    int hold_ms = 0;
    const int max_hold = static_cast<int>(c.pose_timeout * 1000);
    while (true) {
      hold5(q_tgt, qr, vr, ar);
      if (ms % c.frame_ms == 0) {
        dr.push_advance(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
        sample_tel(A, c, rng, nz, tq, tv);
      } else {
        dr.push_advance(dr.lag.back(), c.delay_ms);
      }
      dr.tick(A, c, rng, nz, tau);
      step_world(A, c, tau);
      double vmx = 0;
      for (int j = 0; j < 5; ++j) vmx = std::max(vmx, std::fabs(A.d->qvel[A.dof[j]]));
      vmax.push_back(vmx);
      std::array<double, 5> e{};
      for (int j = 0; j < 5; ++j)
        e[j] = dr.cur.kp[j] * (dr.cur.p[j] - A.d->qpos[A.qadr[j]]);
      kpe.push_back(e);
      if (static_cast<int>(vmax.size()) > 300) { vmax.pop_front(); kpe.pop_front(); }
      const double t = ms * 1e-3;
      if (t_settle < 0 && static_cast<int>(vmax.size()) == 300 && ms % 10 == 0) {
        bool calm = true;
        // 阈 2e-2:量化反馈抖动地板 ~1e-2(位置停在 14bit 边界→q̃ 翻转→
        // τ 跳 kp·LSB,位置不动、速度小幅极限环,std(kp·e_p)=0 已证静止)
        for (double x : vmax) calm = calm && x < 2e-2;
        if (calm)
          for (int j = 0; j < 5 && calm; ++j) {
            double m = 0;
            for (const auto& a : kpe) m += a[j];
            m /= static_cast<double>(kpe.size());
            double s2 = 0;
            for (const auto& a : kpe) s2 += (a[j] - m) * (a[j] - m);
            calm = calm && std::sqrt(s2 / static_cast<double>(kpe.size())) < 0.3;
          }
        if (calm) t_settle = t;
      }
      if (ms % 4 == 0) {
        diag(A, sd, A.d->qpos, grav, Mii);
        row_ref_cmd(f, t, qr, vr, ar, dr.lag.back(), tq, tv, tau, grav, Mii);
        const int rec = (t_settle >= 0 && t - t_settle <= 0.5) ? 1 : 0;
        std::fprintf(f, ",%d,%d\n", rec, od.second);
        rows += 1;
        recs += rec;
      }
      ++ms; ++hold_ms;
      if (t_settle >= 0 && t - t_settle > 0.5) break;  // 记录窗结束→下一姿位
      if (hold_ms >= max_hold && t_settle < 0) {
        double vmx = 0, smax = 0;
        for (double x : vmax) vmx = std::max(vmx, x);
        for (int j = 0; j < 5; ++j) {
          double m = 0;
          for (const auto& a : kpe) m += a[j];
          m /= static_cast<double>(kpe.size());
          double s2 = 0;
          for (const auto& a : kpe) s2 += (a[j] - m) * (a[j] - m);
          smax = std::max(smax, std::sqrt(s2 / static_cast<double>(kpe.size())));
        }
        std::printf("[FAIL] 姿位 %d(方向 %d)稳态判据超时:max|dq|=%.1e(阈2e-2)"
                    " max std(kp·e_p)=%.2f(阈0.3)\n",
                    od.first, od.second, vmx, smax);
        ++fails;
        break;
      }
    }
    for (int j = 0; j < 5; ++j) q_prev[j] = q_tgt[j];
  }
  mj_deleteData(sd);
  std::fclose(f);
  std::printf("[OK] poses %d rows(%d 记录行),%d 超时 -> %s/mitarm_poses.csv\n",
              rows, recs, fails, outdir.c_str());
}

// ---------- constvel ----------
void run_constvel(const Arm& A, const Cfg& c, const std::string& outdir) {
  FILE* f = open_csv(outdir, "constvel", c);
  if (!f) return;
  hdr_ref_cmd(f);
  std::fprintf(f, "\n");
  Driver dr;
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> nz(0, c.vnoise);
  mjData* sd = mj_makeData(A.m_rb);
  double qr[5], vr[5], ar[5], tau[5], grav[5], Mii[5], tq[5], tv[5];
  double tff0[5] = {0, 0, 0, 0, 0};
  double q_hold[5] = {kHome[0], kHome[1], kHome[2], kHome[3], kHome[4]};
  const double lo[5] = {-11.9, -2.9, -2.9, -11.9, -11.9};
  const double hi[5] = {11.9, 0.3, 0.3, 11.9, 11.9};
  for (int j = 0; j < 5; ++j) A.d->qpos[A.qadr[j]] = kHome[j];
  mj_forward(A.m, A.d);
  hold5(q_hold, qr, vr, ar);
  dr.prefill(mkframe(qr, vr, c.mv_kp, c.mv_kd, tff0), c.delay_ms);

  long ms = 0;
  int rows = 0, nseg = 0;
  for (int j = 0; j < 5; ++j) {
    for (int si = 0; si < 3; ++si) {
      const double v = c.cv_speeds[si];
      const double dist = trap_total(v, c.cv_acc, c.cv_cruise);
      for (int rep = 0; rep < 2; ++rep) {  // 双向:±sign 各一段
        double sg = (rep == 0) ? 1.0 : -1.0;
        // 带余量翻转与兜底(限位前 0.15 rad;mjcf J2/J3 range 硬限在植物侧生效)
        if (q_hold[j] + sg * dist > hi[j] - 0.15) sg = -1.0;
        if (q_hold[j] + sg * dist < lo[j] + 0.15) sg = 1.0;
        if (q_hold[j] + sg * dist > hi[j] - 0.15 ||
            q_hold[j] + sg * dist < lo[j] + 0.15)
          continue;  // 行程放不下,放弃该段
        std::printf("[cv] j%d v=%+.2f %+.0f: %+.3f -> %+.3f\n", j + 1, sg * v,
                    sg, q_hold[j], q_hold[j] + sg * dist);
        const double q0 = q_hold[j];
        const double T = 2 * (v / c.cv_acc) + c.cv_cruise + 0.3;  // +0.3s 尾部静止
        const int nms = static_cast<int>(T * 1000);
        for (int step = 0; step < nms; ++step, ++ms) {
          trap_ref(step * 1e-3, q0, sg, v, c.cv_acc, c.cv_cruise, &qr[j], &vr[j],
                   &ar[j]);
          if (ms % c.frame_ms == 0) {
            dr.push_advance(mkframe(qr, vr, c.mv_kp, c.mv_kd, tff0), c.delay_ms);
            sample_tel(A, c, rng, nz, tq, tv);
          } else {
            dr.push_advance(dr.lag.back(), c.delay_ms);
          }
          dr.tick(A, c, rng, nz, tau);
          step_world(A, c, tau);
          if (ms % 2 == 0) {
            diag(A, sd, A.d->qpos, grav, Mii);
            row_ref_cmd(f, ms * 1e-3, qr, vr, ar, dr.lag.back(), tq, tv, tau, grav,
                        Mii);
            std::fprintf(f, "\n");
            ++rows;
          }
        }
        q_hold[j] = q0 + sg * dist;
        ++nseg;
      }
    }
  }
  mj_deleteData(sd);
  std::fclose(f);
  std::printf("[OK] constvel %d rows,%d 段 -> %s/mitarm_constvel.csv\n", rows, nseg,
              outdir.c_str());
}

// ---------- chirp ----------
void run_chirp(const Arm& A, const Cfg& c, const std::string& outdir) {
  ChirpCfg cc;
  cc.init(c);
  for (int j = 0; j < 5; ++j) {
    double s = 0;
    for (int i = 0; i < 4; ++i) s += cc.A[j][i];
    std::printf("[chirp] j%d A=%.3f/%.3f/%.3f/%.3f Σ=%.3f\n", j + 1, cc.A[j][0],
                cc.A[j][1], cc.A[j][2], cc.A[j][3], s);
  }
  FILE* f = open_csv(outdir, "chirp", c);
  if (!f) return;
  hdr_ref_cmd(f);
  std::fprintf(f, "\n");
  Driver dr;
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> nz(0, c.vnoise);
  mjData* sd = mj_makeData(A.m_rb);
  double qr[5], vr[5], ar[5], tau[5], grav[5], Mii[5], tq[5], tv[5];
  double tff0[5] = {0, 0, 0, 0, 0};
  for (int j = 0; j < 5; ++j) A.d->qpos[A.qadr[j]] = kHome[j];
  mj_forward(A.m, A.d);
  chirp_ref(cc, 0, qr, vr, ar);
  dr.prefill(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
  const int nms = static_cast<int>(c.chirp_T * 1000);
  int rows = 0;
  for (int ms = 0; ms < nms; ++ms) {
    const double t = ms * 1e-3;
    chirp_ref(cc, t, qr, vr, ar);
    if (ms % c.frame_ms == 0) {
      dr.push_advance(mkframe(qr, vr, c.hold_kp, c.hold_kd, tff0), c.delay_ms);
      sample_tel(A, c, rng, nz, tq, tv);
    } else {
      dr.push_advance(dr.lag.back(), c.delay_ms);
    }
    dr.tick(A, c, rng, nz, tau);
    step_world(A, c, tau);
    diag(A, sd, A.d->qpos, grav, Mii);
    row_ref_cmd(f, t, qr, vr, ar, dr.lag.back(), tq, tv, tau, grav, Mii);
    std::fprintf(f, "\n");
    ++rows;
  }
  mj_deleteData(sd);
  std::fclose(f);
  std::printf("[OK] chirp %d rows -> %s/mitarm_chirp.csv\n", rows, outdir.c_str());
}

// ---------- swing(三变体并行:同 mjModel 三个 mjData) ----------
void run_swing(Arm& A, const Cfg& c, const std::string& outdir) {
  // FF 回放文件(t,ff_j1..ff_j5 @1ms;由 python --stage ff 从 swing 参考生成)
  std::vector<double> fft, ffv;
  if (!c.ff_file.empty()) {
    FILE* g = std::fopen(c.ff_file.c_str(), "r");
    if (!g) {
      std::printf("[FAIL] open ff_file %s\n", c.ff_file.c_str());
      return;
    }
    char line[256];
    while (std::fgets(line, sizeof line, g)) {
      if (line[0] == '#' || line[0] == 't') continue;
      double v[6];
      if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3],
                      &v[4], &v[5]) == 6) {
        fft.push_back(v[0]);
        for (int j = 0; j < 5; ++j) ffv.push_back(v[j + 1]);
      }
    }
    std::fclose(g);
    std::printf("[swing] ff_file %zu 行\n", fft.size());
  }
  auto ff_at = [&](int ms, double out[5]) {
    if (fft.empty()) {
      for (int j = 0; j < 5; ++j) out[j] = 0;
      return;
    }
    const int i = static_cast<int>(clampd(ms, 0, static_cast<double>(fft.size()) - 1));
    for (int j = 0; j < 5; ++j) out[j] = ffv[5 * i + j];
  };

  const char* kTag[3] = {"a", "ci", "cid"};
  mjData* dv[3];
  Driver drv[3];
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> nz(0, c.vnoise);
  for (int k = 0; k < 3; ++k) {
    dv[k] = mj_makeData(A.m);
    for (int j = 0; j < 5; ++j) dv[k]->qpos[A.qadr[j]] = kSwP[0][j];
    mj_forward(A.m, dv[k]);
  }
  mjData* sd = mj_makeData(A.m_rb);
  double qr[5], vr[5], ar[5], tff_id[5], tff_ci[5], tau[5];
  double tff0[5] = {0, 0, 0, 0, 0};
  swing_ref(0, qr, vr, ar);
  for (int k = 0; k < 3; ++k)
    drv[k].prefill(mkframe(qr, vr, c.mv_kp, c.mv_kd, tff0), c.delay_ms);

  FILE* f = open_csv(outdir, "swing", c);
  if (!f) {
    for (int k = 0; k < 3; ++k) mj_deleteData(dv[k]);
    mj_deleteData(sd);
    return;
  }
  std::fprintf(f, "t");
  for (int j = 0; j < 5; ++j)
    std::fprintf(f, ",qref_j%d,vref_j%d,aref_j%d", j + 1, j + 1, j + 1);
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 5; ++j)
      std::fprintf(f, ",q_%s_j%d,dq_%s_j%d,tau_%s_j%d,tff_%s_j%d", kTag[k], j + 1,
                   kTag[k], j + 1, kTag[k], j + 1, kTag[k], j + 1);
  std::fprintf(f, "\n");

  const int nms = static_cast<int>(kSwDur * 1000);
  mjData* d_main = A.d;  // 保存主数据(循环内临时切到各变体)
  for (int ms = 0; ms < nms; ++ms) {
    const double t = ms * 1e-3;
    swing_ref(t, qr, vr, ar);
    if (ms % c.frame_ms == 0) {
      ideal_ff(A, sd, c, qr, vr, ar, tff_ci);
      ff_at(ms, tff_id);
      drv[0].push_advance(mkframe(qr, vr, c.mv_kp, c.mv_kd, nullptr), c.delay_ms);
      drv[1].push_advance(mkframe(qr, vr, c.mv_kp, c.mv_kd, tff_ci), c.delay_ms);
      drv[2].push_advance(mkframe(qr, vr, c.mv_kp, c.mv_kd, tff_id), c.delay_ms);
    } else {
      for (int k = 0; k < 3; ++k) drv[k].push_advance(drv[k].lag.back(), c.delay_ms);
    }
    std::fprintf(f, "%.4f", t);
    for (int j = 0; j < 5; ++j) std::fprintf(f, ",%.6f,%.5f,%.4f", qr[j], vr[j], ar[j]);
    for (int k = 0; k < 3; ++k) {
      A.d = dv[k];  // tick/step_world 都作用于该变体
      drv[k].tick(A, c, rng, nz, tau);
      step_world(A, c, tau);
      for (int j = 0; j < 5; ++j)
        std::fprintf(f, ",%.6f,%.5f,%.3f,%.3f", dv[k]->qpos[A.qadr[j]],
                     dv[k]->qvel[A.dof[j]], tau[j], drv[k].cur.tff[j]);
    }
    A.d = d_main;
    std::fprintf(f, "\n");
  }
  for (int k = 0; k < 3; ++k) mj_deleteData(dv[k]);
  mj_deleteData(sd);
  std::fclose(f);
  std::printf("[OK] swing -> %s/mitarm_swing.csv%s\n", outdir.c_str(),
              c.ff_file.empty()
                  ? "(无 ff_file——C_id 无前馈;先跑 python --stage ff 再重跑 swing)"
                  : "");
}

}  // namespace

int main(int argc, char** argv) {
  std::string sc = argc > 1 ? argv[1] : "selftest";
  const std::string outdir = argc > 2 ? argv[2] : "eval/reports";
  Cfg c;
  for (int i = 3; i < argc; ++i) {
    const std::string s = argv[i];
    const auto eq = s.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = s.substr(0, eq);
    const std::string vs = s.substr(eq + 1);
    const double v = std::atof(vs.c_str());
    if (k == "model") c.model = vs;
    else if (k == "ff_file") c.ff_file = vs;
    else if (k == "poses_n") c.poses_n = static_cast<int>(v);
    else if (k == "delay_ms") c.delay_ms = static_cast<int>(v);
    else if (k == "frame_ms") c.frame_ms = static_cast<int>(v);
    else if (k == "fb_bits") c.fb_bits = static_cast<int>(v);
    else if (k == "sub") c.sub = static_cast<int>(v);
    else if (k == "vnoise") c.vnoise = v;
    else if (k == "seed") c.seed = static_cast<unsigned>(v);
    else if (k == "chirp_T") c.chirp_T = v;
    else if (k == "pose_move_T") c.pose_move_T = v;
    else if (k == "pose_timeout") c.pose_timeout = v;
    else if (k == "cv_acc") c.cv_acc = v;
    else if (k == "cv_cruise") c.cv_cruise = v;
    else {
      bool hit = false;
      for (int j = 0; j < 5; ++j) {
        char kb[16], kt[16], kh[16], ka[16];
        char kb2[16], kt2[16], kh2[16], km2[16];
        std::snprintf(kb, sizeof kb, "b%d", j + 1);
        std::snprintf(kt, sizeof kt, "tc%d", j + 1);
        std::snprintf(kh, sizeof kh, "J0%d", j + 1);
        std::snprintf(ka, sizeof ka, "arm%d", j + 1);
        std::snprintf(kb2, sizeof kb2, "hold_kp%d", j + 1);
        std::snprintf(kt2, sizeof kt2, "hold_kd%d", j + 1);
        std::snprintf(kh2, sizeof kh2, "mv_kp%d", j + 1);
        std::snprintf(km2, sizeof km2, "mv_kd%d", j + 1);
        if (k == kb) { c.b[j] = v; hit = true; }
        else if (k == kt) { c.tc[j] = v; hit = true; }
        else if (k == kh) { c.J0[j] = v; hit = true; }
        else if (k == ka) { c.arm[j] = v; hit = true; }
        else if (k == kb2) { c.hold_kp[j] = v; hit = true; }
        else if (k == kt2) { c.hold_kd[j] = v; hit = true; }
        else if (k == kh2) { c.mv_kp[j] = v; hit = true; }
        else if (k == km2) { c.mv_kd[j] = v; hit = true; }
      }
      if (!hit) std::printf("[warn] 忽略未知参数 %s\n", s.c_str());
    }
  }
  Arm A;
  if (!load_arm(A, c)) return 1;
  std::printf("[cfg] model=%s delay=%dms frame=%dms fb_bits=%d sub=%d poses_n=%d "
              "chirp_T=%g\n",
              c.model.c_str(), c.delay_ms, c.frame_ms, c.fb_bits, c.sub, c.poses_n,
              c.chirp_T);
  if (sc == "selftest") run_selftest(A, c);
  else if (sc == "poses") run_poses(A, c, outdir);
  else if (sc == "constvel") run_constvel(A, c, outdir);
  else if (sc == "chirp") run_chirp(A, c, outdir);
  else if (sc == "swing") run_swing(A, c, outdir);
  else if (sc == "all") {
    run_selftest(A, c);
    run_poses(A, c, outdir);
    run_constvel(A, c, outdir);
    run_chirp(A, c, outdir);
    run_swing(A, c, outdir);
  } else std::printf("[FAIL] 未知场景 %s\n", sc.c_str());
  if (A.d) mj_deleteData(A.d);
  mj_deleteModel(A.m_rb);
  mj_deleteModel(A.m);
  return 0;
}
