#pragma once
// 旧仓库实机 NMPC 独立模块化(badminton-mpc-6.0,2026-10-09 移植)。
//
// 【来源与分工】原 nmpc.{h,cc} 在实机羽毛球机器人上验证可用,拆两层:
//   ① 生成码层 third_party/nmpc6/nlp.so —— CasADi 3.7 codegen 的 OCP 数值
//      函数库(f/g/梯度/雅可比/黑塞全静态编入,ldd 仅 libm/libc),由旧仓库
//      set_my_nmpc_solver() 的 generate_dependencies("nlp.c") 产出,与本模块
//      的公式/布局强绑定(改公式须重生成,见 third_party/nmpc6/README)。
//   ② 本驱动层 —— casadi::nlpsol 动态加载 nlp.so(IPOPT),负责参数/上下界
//      装配、数值 rollout、守门链、回合状态。OCP 构型:
//      14 态 [x,y,θ1..5,vx,vy,ω1..5],7 控 [dvx,dvy,dω1..5](速度增量,梯形
//      积分含刹车距离修正),3 步预测;代价=终端唯一(法向对准 1000 + 位置
//      2500 + 速度误差 + 控制量小罚);约束=逐关节动态角度窗(前场/正常球
//      分档)+速度窗+加速度 |U|≤a_max·dt+末端 z≥0.35。
//
// 【剥离清单(相对原文件)】boost::filesystem(日志目录)、OpenCV(TickMeter→
// chrono)、share/util 时间戳、ONNX 误差网×2 → 可注入回调 error_comp(默认
// 空=恒零;真机重挂 ViconErrorNetInfer)、master/constant.h、硬编码
// /home/x4-04 路径 → LegacyNmpcConfig。MpcData 落盘与 SetMpcStates 互斥队列
// 删(日志/状态由调用方自取 Result)。
//
// 【守门链(实机验证顺序,全部保留)】新回合检测(全零 guess)→ 误差补偿注入
// (剩余时间≤50ms 锁存)→ 冻结窗(剩余>1s 且被冻关节静止 → 首步 dω4/dω5
// lbx=ubx=0)→ 求解 → 梯形 rollout → 1ms 步自碰复检(FK z<0.35)→ 解跳变
// 守门(m1>60°/m3>50°/m4>90°)→ 角度窗+10° 守门 → 精度门(严格 0.05m 三轴
// +|vx|>0.6·hit_spd / 宽松 0.20m+求解成功双档)。拒绝全部显式化进
// LegacyNmpcStatus(原代码统一返回空 vector,排障靠 cout)。
//
// 【移植注】线性求解器默认 mumps(随 casadi 便携包,免 HSL);原机用 MA57
// (libcoinhsl),config 可切回。手写速度雅可比省略 ω2/ω5 项 —— 原机由误差
// 网吸收,sim 下是模型误差的一部分,保留原样(与生成码一致,不可单方改)。
#include <functional>
#include <string>
#include <vector>

namespace sckd {

struct LegacyNmpcConfig {
  std::string nlp_so = "third_party/nmpc6/nlp.so";  // 生成码(相对路径按
                                                    // 进程工作目录解析)
  double racket_len = 0.85;                         // d5=−racket_len(X1 机型)
  std::string linear_solver = "mumps";              // mumps / ma57(需 libcoinhsl)
  bool sim_joint23_limits = false;  // 仿真关节坐标直接来自加载的 MJCF
  double joint23_lo[2] = {-3.14, -3.14}, joint23_hi[2] = {0.5, 0.5};
  bool actuator_prediction = false;
  bool enforce_feasibility = false; // 新候选：终端锚、状态/约束/精度联合验收
  double constraint_tolerance = 1e-4;
  double terminal_tolerance_m = .05;
  int max_iter = 270;                               // IPOPT 上限(原值)
};

struct LegacyNmpcRequest {
  // 状态:[x,y,θ1..θ5, vx,vy, ω1..ω5](m/rad, m/s / rad/s;θ 为关节角)
  double state[14] = {0};
  double target[3] = {0};      // 击球点(m,世界系)
  double dt[3] = {0};          // 每步时长(s;和=剩余击球时间,均分由调用方给)
  double effective_q[5]={0},effective_dq[5]={0};
  double hit_spd = 10.0;       // 出球速度 m/s(拍末端 vx 目标)
  double hit_yaw_deg = 0.0;    // 出球方位(°)
  double hit_pitch_deg = 20.0; // 出球仰角(°,决定拍面法向目标)
  double hit_speed_z = 2.0;    // 拍末端 vz 目标(m/s)
  // 底盘锚邻域(可选,2026-10-09 m2 接线引入):开启后状态约束里的底盘 x/y
  // 行从全场 [−1,5.7]×[−2,2] 收紧到锚 ±r(交集)。旧机无此约束 —— 其可达性
  // 由上游走位策略保证;sim 侧同源物 = 规划器 base_target(IK 可达环带)。
  // 不加时 NMPC 自由落位出现过"底盘停 1.3m 外臂全伸展凑目标",末段目标
  // 精化外移 → 全拒收 → 冻结旧计划,全伸展下关节小误差拍端放大 0.36m miss
  bool corridor_ok=false;double corridor_xy[3][2]={{0}};double corridor_radius=.25;
  bool base_anchor_ok = false;
  double base_anchor[2] = {0, 0};
  double base_anchor_r = 0.40;
};

enum class LegacyNmpcStatus {
  kOk = 0,         // 严格精度档(三轴<thr 且 |vx|>0.6·hit_spd;可执行)
  kLoose,          // 宽松档(三轴<0.20m 且 IPOPT 成功;可执行)
  kSolverFail,     // IPOPT 未返回成功解(结果不可用)
  kSelfCollision,  // 1ms 复检:末端 z<0.35(挥拍路径打机器/地面)
  kJump,           // 与上一接受解跳变过大(m1>60°/m3>50°/m4>90°)
  kAngleLimit,     // rollout 终态角度超窗(动态角窗+10°)
  kAccFail,        // 精度门未过且不满足宽松档(解不可用)
  kConstraintViolation, kInputInvalid, kTerminalError,
};

struct LegacyNmpcResult {
  LegacyNmpcStatus status = LegacyNmpcStatus::kAccFail;
  double max_constraint_violation = 0;
  bool solver_ok = false;      // IPOPT return_status ∈ {Solved, Acceptable}
  double term_state[14] = {0}; // rollout 终态=下发目标(x/y 含末端误差反馈修正)
  double controls[21] = {0};   // 控制增量,块序 [dvx×3, dvy×3, dω1×3 … dω5×3]
                               // (与生成码决策变量布局一致;交错序转换自做)
  double end_pos[7] = {0};     // 终态末端:[px,py,pz,pitch°,vx,vy,vz]
  double pred_vxy_at_hit[2] = {0};  // 首解捕获的触球底盘速度(后续重解不更新)
  double solve_ms = 0;
  // 1ms 执行轨迹(Plans 式,robot_motion.cc MpcThread 同口径 —— 旧下位机实际
  // 消费的就是它):从求解时刻实测状态起,速度按 a_max 限幅斜坡趋近逐步累积
  // 目标速度 v_cmd[j]=v0+Σ_{k≤j}u_k,位置欧拉积分,跨步连续。网格 i=0..plan_ms,
  // 布局 [c*(plan_ms+1)+i],c=0..6 = [x,y,θ1..θ5]。旧机按剩余时间索引;sim
  // 侧按绝对时刻(求解时刻 + i ms)采样。
  int plan_ms = 0;
  std::vector<double> plan_pos;   // 7*(plan_ms+1)
  std::vector<double> plan_vel;
};

class LegacyNmpc {
 public:
  using ErrorComp = std::function<void(
      const double end_pos[7],        // 本次预测末端(px,py,pz,pitch°,v*3)
      double comp[5])>;               // 出 [Δx,Δy,Δz,Δpitch°,Δyaw°](原始值;
                                     // 注入与锁存/限幅同原实现)
  explicit LegacyNmpc(const LegacyNmpcConfig& cfg = LegacyNmpcConfig());
  ~LegacyNmpc();

  // 新回合:清 guess/回合状态(等价 SetGuss;全零 guess 时 Solve 内部也会
  // 自动触发同一重置 —— 原实现语义,双保险)
  void Reset();

  // 求解(每拍调用,dt 随剩余时间收缩)。false=不可执行(status 给原因);
  // true=status∈{kOk,kLoose},controls/term_state 可下发。
  bool Solve(const LegacyNmpcRequest& req, LegacyNmpcResult* out);

  // 外部播种初始猜测(块序 21;等价 setInitGuss_Order)
  void SetGuess(const double guess_block[21]);

  ErrorComp error_comp;   // 误差补偿注入点(默认空=恒零补偿)

  // 诊断(实例级累计)
  int solves = 0, rejects = 0;
  double ms_max = 0, ms_sum = 0;

 private:
  struct Impl;
  Impl* impl_ = nullptr;   // casadi 类型不进头(去依赖传播)
};

}  // namespace sckd
