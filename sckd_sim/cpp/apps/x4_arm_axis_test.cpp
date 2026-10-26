// 只使用指定 SDK 的机械臂接口；默认 inspect，不创建任何设备服务。
#include "runtime/arm_axis_check.h"
#include "runtime/arm_startup_check.h"
#include "runtime/x4_arm_test_backend.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace ac = sckd::io::axischeck;
namespace fs = std::filesystem;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Signal(int) { interrupted = 1; }
double Now() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) throw std::runtime_error("clock_gettime failed");
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
void SleepUntil(double time) {
  timespec ts{static_cast<time_t>(time), static_cast<long>((time - std::floor(time)) * 1e9)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR && !interrupted) {}
}
std::string Quote(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
template <class Values> std::string Array(const Values& values) {
  std::ostringstream out; out << std::setprecision(17) << std::boolalpha << '[';
  bool first = true;
  for (const auto& value : values) {
    if (!first) out << ',';
    first = false;
    if constexpr (std::is_floating_point_v<std::decay_t<decltype(value)>>) {
      if (!std::isfinite(value)) { out << "null"; continue; }
    }
    out << value;
  }
  out << ']'; return out.str();
}
struct Options {
  std::string backend = "inspect", config, output;
  std::string test_mode = "direction";
  int step_deg = 30;
  bool demo = false, ui = false;
};
ac::Policy MotionPolicy(const Options& o) {
  ac::Policy policy;
  policy.direction_only = o.test_mode == "direction";
  if(policy.direction_only) policy.max_feedback_excursion_rad = 45 * ac::kDeg;
  policy.step_rad = o.step_deg * ac::kDeg;
  // 用户指定命令累计限制45°，单次30°不能叠加到60°。
  policy.max_excursion_rad = 45 * ac::kDeg;
  return policy;
}
void Help() {
  std::cout << "X4 机械臂逐轴测试（J1..J5，SDK模型角）\n"
    "--config FILE [--backend inspect|sim|hardware] [--output NEW_DIRECTORY] [--step-deg 1|10|30] [--test-mode direction|tracking] [--demo] [--ui]\n"
    "默认 inspect：只读配置；sim：SDK机械臂流程仿真；hardware：会使能并到配置home。\n"
    "--demo 仅允许sim，自动执行每轴正向一步再反向返回；不生成实物方向确认。\n"
    "交互命令：status / + / - / loop N / pause / stop / next / select N / record N same|opposite 观察说明 / quit\n"
    "loop N：当前轴正向一步再返回原点，重复1..10次，每个端点稳定后停0.5秒。pause完成当前往返后暂停。\n"
    "--ui：每20ms输出X4_STATE JSON，初始化后须每500ms内收到heartbeat；与--demo互斥。\n"
    "默认每次30°，直接下发位置目标，由SDK/电机执行；命令累计限制±45°，不允许连续加到60°。\n"
    "可用--step-deg 1或10选择较小步长；命令累计限制仍为±45°，需五轴位置稳定后才允许下一步。\n"
    "默认direction只测反馈增量方向，忽略小角度残差/联动；tracking恢复原精确残差检查。\n"
    "record：在正向一步完成后填写实际运动的物理关节号及方向。仅记录，不改SDK配置。\n"
    "stop终止本轮点动并冻结目标，之后仅可status/help/quit；quit/SIGINT调用SDK Stop。均不是急停。\n"
    "每批只输入一条非heartbeat命令；多条拒绝整批动作，quit/stop/pause优先。\n";
}
Options Parse(int argc, char** argv) {
  Options o;
  for (int i=1;i<argc;++i) {
    const std::string key=argv[i];
    if (key=="--demo") o.demo=true;
    else if (key=="--ui") o.ui=true;
    else if (key=="--step-deg") {
      if (++i==argc) throw std::invalid_argument("missing value: "+key);
      const std::string value=argv[i];
      if(value!="1" && value!="10" && value!="30") throw std::invalid_argument("--step-deg must be 1, 10 or 30");
      o.step_deg=std::stoi(value);
    }
    else if (key=="--backend" || key=="--config" || key=="--output" || key=="--test-mode") {
      if (++i==argc) throw std::invalid_argument("missing value: "+key);
      (key=="--backend" ? o.backend : key=="--config" ? o.config : key=="--output" ? o.output : o.test_mode)=argv[i];
    } else throw std::invalid_argument("unknown option: "+key);
  }
  if (o.config.empty()) throw std::invalid_argument("--config is required");
  if (o.backend!="inspect" && o.backend!="sim" && o.backend!="hardware")
    throw std::invalid_argument("unknown backend");
  if(o.test_mode!="direction" && o.test_mode!="tracking")
    throw std::invalid_argument("--test-mode must be direction or tracking");
  if (o.demo && o.backend!="sim") throw std::invalid_argument("--demo is allowed only with sim");
  if (o.demo && o.ui) throw std::invalid_argument("--ui and --demo cannot be combined");
  if (o.backend!="inspect" && o.output.empty()) throw std::invalid_argument("--output NEW_DIRECTORY is required");
  if (o.backend=="hardware" && (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)))
    throw std::invalid_argument("hardware requires an interactive terminal, not pipes or command files");
  return o;
}
struct Record { bool set=false; int physical=0; std::string direction,note; };
struct UiState {
  bool enabled=false, has_ref=false, session_stopped=false, loop_active=false;
  int selected=0, loop_remaining=0;
  uint64_t seq=0;
  double last_emit=-1;
  std::string backend, phase="not_started";
  std::string test_mode;
  bool has_anchor=false;
  ac::Joints command_origin{}, feedback_anchor{};
  bool feedback_observed=false;
  ac::Output submitted{}, status{};
  ac::Feedback feedback{};
  void Ready(const char* phase) const {
    if (!enabled) return;
    std::cout << "\nX4_UI_READY {\"schema\":\"X4_ARM_COMPARE/1\",\"backend\":" << Quote(backend)
      << ",\"phase\":" << Quote(phase)
      << ",\"heartbeat_timeout_s\":0.5,\"state_period_s\":0.02}\n" << std::flush;
  }
  void Emit(double now, bool force=false) {
    if (!enabled || (!force && last_emit>=0 && now-last_emit<.02)) return;
    last_emit=now;
    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha
      << "{\"schema\":\"X4_ARM_COMPARE/1\",\"seq\":" << seq++
      << ",\"t_monotonic\":" << now << ",\"backend\":" << Quote(backend)
      << ",\"phase\":" << Quote(phase) << ",\"feedback_observed\":" << feedback_observed
      << ",\"test_mode\":" << Quote(test_mode)
      << ",\"selected_joint\":" << selected+1
      << ",\"q_ref\":" << (has_ref ? Array(submitted.q) : "[null,null,null,null,null]")
      << ",\"dq_ref\":" << (has_ref ? Array(submitted.dq) : "[null,null,null,null,null]")
      << ",\"q_fb\":" << Array(feedback.sample.q) << ",\"dq_fb\":" << Array(feedback.sample.dq)
      << ",\"stamp_s\":" << Array(feedback.sample.stamp_s)
      << ",\"fault_bits\":" << Array(feedback.sample.fault_bits)
      << ",\"mode_state\":" << Array(feedback.sample.mode_state)
      << ",\"raw_mode_state\":" << Array(feedback.raw_mode_state)
      << ",\"mode_state_synthetic\":" << feedback.mode_state_synthetic
      << ",\"can_open\":" << Array(feedback.can_open)
      << ",\"initialized\":" << status.initialized << ",\"moving\":" << status.moving
      << ",\"settled\":" << status.settled << ",\"fault\":" << status.fault
      << ",\"session_stopped\":" << session_stopped << ",\"reason\":" << Quote(status.reason)
      << ",\"loop_active\":" << loop_active << ",\"loop_remaining\":" << loop_remaining
      << ",\"completed_move_id\":" << status.completed_move_id
      << ",\"completed_joint\":" << status.completed_joint+1
      << ",\"command_delta\":" << status.completed_command_delta
      << ",\"feedback_delta\":" << Array(status.feedback_delta)
      << ",\"response_direction\":" << Quote(status.response_direction)
      << ",\"snapshot_attempts\":" << feedback.snapshot_attempts;
    if(has_anchor) out << ",\"command_origin\":" << Array(command_origin)
                      << ",\"feedback_anchor\":" << Array(feedback_anchor);
    out << '}';
    std::cout << "X4_STATE " << out.str() << '\n' << std::flush;
  }
  void Terminal(const std::string& reason, bool fault) {
    status.reason=reason; status.fault=fault; status.moving=false; status.settled=false;
    session_stopped=true; loop_active=false; loop_remaining=0;
    Emit(Now(),true);
  }
};
struct StopGuard {
  ac::SdkArmBackend& backend;
  UiState* ui=nullptr;
  ac::Output last{};
  bool sent=false, closed=false;
  void Close() {
    if(closed) return;
    closed=true;
    // 日志/输入异常也先冻结最后成功提交的位置；不根据失效反馈重设目标。
    if(sent) {
      auto hold=last; hold.dq.fill(0); std::string ignored;
      if (backend.Write(hold,&ignored)) {
        last=hold;
        if (ui) { ui->submitted=hold; ui->has_ref=true; }
      }
    }
    backend.Stop();
  }
  ~StopGuard() { try { if (!closed) { Close(); if(ui) ui->Terminal("exception_or_start_failure",true); } } catch(...) {} }
};
struct LoopPlan {
  enum class Stage { ready, positive, positive_dwell, returning, origin_dwell };
  bool active=false, pause=false;
  int remaining=0;
  double since=0;
  Stage stage=Stage::ready;
  void Clear() { active=false; pause=false; remaining=0; }
};
bool AtOrigin(const ac::Controller& c) {
  for (int j=0;j<5;++j) if (std::abs(c.status().q[j]-c.origin()[j])>1e-8) return false;
  return true;
}
void Status(const ac::Controller& c, const ac::Sample& s, int joint) {
  std::cout << "J" << joint+1 << " " << c.status().reason
    << " moving=" << c.status().moving << " settled=" << c.status().settled << "\n";
  std::cout << "反馈deg / 目标deg:";
  for (int j=0;j<5;++j) std::cout << " J" << j+1 << '=' << std::fixed << std::setprecision(2)
    << s.q[j]/ac::kDeg << '/' << c.status().q[j]/ac::kDeg;
  std::cout << '\n' << std::flush;
}
bool Healthy(const ac::Feedback& f, double now, std::string& why) {
  if (!f.running) { why="server_not_running"; return false; }
  if (!std::all_of(f.can_open.begin(),f.can_open.end(),[](bool x){return x;})) {
    why="missing_can_channel"; return false;
  }
  for (int j=0;j<5;++j) {
    if (!std::isfinite(f.sample.q[j]) || !std::isfinite(f.sample.dq[j]) ||
        !std::isfinite(f.torque[j]) || !std::isfinite(f.temperature[j]) ||
        f.sample.stamp_s[j]<=0 || f.sample.stamp_s[j]>now || now-f.sample.stamp_s[j]>.10 ||
        f.sample.fault_bits[j]!=0 || f.sample.mode_state[j]!=2) {
      why="invalid_feedback:J"+std::to_string(j+1); return false;
    }
  }
  return true;
}
// 排空当前到达的输入再处理；不把动作逐行积压到未来的ready周期。
std::vector<std::string> InputBatch(std::string& pending, bool& eof) {
  while(!eof) {
    pollfd p{STDIN_FILENO,POLLIN,0};
    const int result=poll(&p,1,0);
    if (result<0 && errno!=EINTR) throw std::runtime_error("stdin poll failed");
    if (result>0 && (p.revents&(POLLIN|POLLHUP))) {
      char buf[256]; const auto n=read(STDIN_FILENO,buf,sizeof(buf));
      if (n==0) eof=true;
      else if (n>0) pending.append(buf,static_cast<size_t>(n));
      else if (errno!=EINTR && errno!=EAGAIN) throw std::runtime_error("stdin read failed");
    } else break;
    if (pending.size()>4096) throw std::runtime_error("input batch too long; stopping session");
  }
  std::vector<std::string> lines;
  for(auto end=pending.find('\n');end!=std::string::npos;end=pending.find('\n')) {
    std::string line=pending.substr(0,end); pending.erase(0,end+1);
    if (!line.empty() && line.back()=='\r') line.pop_back();
    if(line.find_first_not_of(" \t")!=std::string::npos) lines.push_back(line);
  }
  return lines;
}
void Metadata(std::ostream& out, const Options& o, const ac::Setup& s) {
  const auto policy=MotionPolicy(o);
  out << "{\n\"schema\":\"X4_ARM_AXIS_TEST/1\",\"backend\":" << Quote(o.backend)
    << ",\"sdk_root\":" << Quote(X4_ARM_SDK_ROOT) << ",\"config\":" << Quote(fs::absolute(o.config).string())
    << ",\"frame\":\"sdk_model\",\"position_unit\":\"rad\",\"clock\":\"CLOCK_MONOTONIC\","
    << "\"sdk_target_handoff\":\"validated_write_then_enable\","
    << "\"test_mode\":" << Quote(o.test_mode)
    << ",\"angular_deviation_policy\":" << Quote(policy.direction_only ? "diagnostic_only_with_fixed_excursion" : "tracking_and_other_axis_fault")
    << ",\"snapshot_max_attempts\":3,"
    << "\"step_deg\":" << o.step_deg << ",\"peak_speed_deg_s\":null,\"command_profile\":\"direct_position\",\"excursion_deg\":"
    << policy.max_excursion_rad/ac::kDeg << ",\"period_s\":" << s.period_s
    << ",\"feedback_excursion_deg\":" << (policy.max_feedback_excursion_rad>0 ? policy.max_feedback_excursion_rad : policy.max_excursion_rad)/ac::kDeg
    << ",\"home\":" << Array(s.home) << ",\"kp\":" << Array(s.kp) << ",\"kd\":" << Array(s.kd)
    << ",\"kp_exit\":" << Array(s.kp_exit) << ",\"kd_exit\":" << Array(s.kd_exit)
    << ",\"joint_sign\":" << Array(s.sign) << ",\"joint_offset\":" << Array(s.offset)
    << ",\"dims\":" << Array(s.dims)
    << ",\"can_if\":[" << Quote(s.can_if[0]) << ',' << Quote(s.can_if[1]) << ',' << Quote(s.can_if[2]) << ']'
    << ",\"simulation_profile\":\"workflow_only_no_gravity_no_noise_9ms_delay\","
    << "\"simulation_profile_applied\":" << (o.backend=="sim" ? "true" : "false")
    << ",\"sim_inertia\":" << Array(s.sim_inertia) << ",\"sim_damping\":" << Array(s.sim_damping)
    << ",\"sim_gravity_amplitude\":" << Array(s.sim_gravity_amplitude)
    << ",\"sim_noise_deg\":" << s.sim_noise_deg << ",\"sim_command_delay_ms\":" << s.sim_command_delay_ms
    << ",\"sim_floor_check\":" << (s.sim_floor_check ? "true" : "false") << ','
    << "\"sim_preseed_home\":" << (o.backend=="sim" && s.sim_preseed_home ? "true" : "false") << ','
    << "\"simulation_mode_state_synthetic\":" << (o.backend=="sim" ? "true" : "false")
    << ",\"feedback_freshness_s\":" << policy.freshness_s
    << ",\"startup_max_tracking_deg\":" << policy.startup_max_tracking_rad/ac::kDeg
    << ",\"max_tracking_deg\":" << policy.max_tracking_rad/ac::kDeg
    << ",\"max_feedback_speed_deg_s\":" << policy.max_feedback_speed_rad_s/ac::kDeg
    << ",\"max_control_gap_s\":" << policy.max_tick_interval_s
    << ",\"settle_time_s\":" << policy.settle_time_s
    << ",\"settle_mode\":\"position_window\",\"settle_span_rad\":" << policy.settle_span_rad
    << ",\"settle_error_rad\":null,\"settle_speed_rad_s\":null,"
    << "\"simulation_is_hardware_validation\":false,\"zero_calibrated\":false}\n";
}
int Run(const Options& o) {
  ac::SdkArmBackend backend(o.config);  // 配置解析；尚未构造硬件服务。
  const auto& setup=backend.setup();
  if(setup.period_s<.001 || setup.period_s>.010)
    throw std::invalid_argument("test tool requires control_period_ms within 1..10 ms");
  Metadata(std::cout,o,setup);
  if (o.backend=="inspect") return 0;
  if (fs::exists(o.output)) throw std::invalid_argument("output exists; choose a new directory");
  UiState ui; ui.enabled=o.ui; ui.backend=o.backend; ui.test_mode=o.test_mode;
  if (o.backend=="hardware") {
    ui.Ready("awaiting_start_home");
    std::cout << "启动前确认：SDK Start 会使能机械臂并直接向上面的home目标运动，"
      "初始化没有本工具的单步斜坡限制；必须已完成CAN配置、排除其他主站，"
      "并确认现场姿态与home、运动范围和急停条件。\n"
      "SDK Stop仍会发送终末帧且不失能；本工具不会启动底盘。\n"
      "现场确认后输入 START-HOME，其他输入取消：" << std::flush;
    std::string confirmation; char ch=0;
    while(!interrupted && confirmation.size()<80) {
      const auto n=read(STDIN_FILENO,&ch,1);
      if(n<0 && errno==EINTR) continue;
      if(n!=1) return 2;
      if(ch=='\n') break;
      if(ch!='\r') confirmation+=ch;
    }
    if(confirmation!="START-HOME" || interrupted) return 2;
  }
  if (!fs::create_directory(o.output)) throw std::runtime_error("cannot create new output directory");
  auto Open=[&](const char* name) {
    std::ofstream f(fs::path(o.output)/name);
    f.exceptions(std::ios::badbit|std::ios::failbit); f << std::setprecision(17); return f;
  };
  auto meta=Open("manifest.json"); Metadata(meta,o,setup); meta.close();
  auto events=Open("events.jsonl");
  auto csv=Open("samples.csv");
  csv << "monotonic_s,selected_joint,moving,settled,fault,mode_state_synthetic";
  for(int j=1;j<=5;++j) for(const char* field:{"q","dq","target_q","target_dq","stamp_s","fault_bits","mode","raw_mode","torque","temperature"})
    csv << ",j" << j << '_' << field;
  csv << '\n';
  auto Event=[&](const std::string& type,const std::string& detail) {
    events << "{\"t\":" << Now() << ",\"type\":" << Quote(type) << ",\"detail\":" << Quote(detail) << "}\n";
  };
  auto startup_log=Open("startup_feedback.jsonl");
  auto observations=Open("motion_observations.jsonl");
  auto Snapshot=[&](const std::string& phase, const std::string& reason) {
    const auto& f=ui.feedback;
    startup_log << "{\"t\":" << Now() << ",\"phase\":" << Quote(phase)
      << ",\"reason\":" << Quote(reason) << ",\"feedback_observed\":" << (ui.feedback_observed ? "true":"false")
      << ",\"q\":" << Array(f.sample.q) << ",\"dq\":" << Array(f.sample.dq)
      << ",\"target_q\":" << Array(backend.LastTarget()) << ",\"stamp_s\":" << Array(f.sample.stamp_s)
      << ",\"fault_bits\":" << Array(f.sample.fault_bits) << ",\"mode_state\":" << Array(f.sample.mode_state)
      << ",\"raw_mode_state\":" << Array(f.raw_mode_state) << ",\"torque\":" << Array(f.torque)
      << ",\"temperature\":" << Array(f.temperature) << ",\"can_open\":" << Array(f.can_open)
      << ",\"running\":" << (f.running ? "true":"false") << "}\n" << std::flush;
  };
  StopGuard stop_guard{backend,&ui};
  try {
  std::string error;
  Event("starting",o.backend);
  if(o.backend=="sim") fs::create_directory(fs::path(o.output)/"sdk_sim");
  if(interrupted) { Event("cancelled_before_start","signal"); return 130; }
  ui.phase="sdk_start";
  Event("sdk_start", "blocking SDK call; no application feedback or heartbeat supervision until return");
  const bool started=o.backend=="sim" ? backend.StartSim((fs::path(o.output)/"sdk_sim").string(),&error) : backend.StartHardware(&error);
  ui.feedback=backend.Read(); ui.feedback_observed=true;
  Snapshot("sdk_start_returned",started ? "success" : error);
  if (!started) throw std::runtime_error("SDK Start failed: "+error);
  const double start=Now();
  const auto policy=MotionPolicy(o);
  ac::StartupCheck startup(start,backend.LastTarget(),policy);
  ui.phase="startup_settling"; ui.Ready("startup_settling");
  ui.submitted.q=backend.LastTarget(); ui.has_ref=true;
  double startup_heartbeat=start;
  std::string startup_pending;
  bool startup_eof=false;
  ac::Feedback feedback{};
  do {
    feedback=backend.Read(); ui.feedback=feedback;
    const double now=Now();
    bool cancel=false;
    if(!o.demo) for(const auto& line:InputBatch(startup_pending,startup_eof)) {
      if(line=="heartbeat") startup_heartbeat=now;
      else if(line=="quit" || line=="stop") cancel=true;
      else Event("startup_command_rejected",line);
    }
    std::string reason;
    if(interrupted || cancel || startup_eof) reason="startup_cancelled";
    else if(o.ui && now-startup_heartbeat>.5) reason="startup_heartbeat_timeout";
    else if(o.backend=="sim" && now-start<.1 &&
            std::all_of(feedback.sample.stamp_s.begin(),feedback.sample.stamp_s.end(),[](double stamp){return stamp==0;})) {
      Snapshot("sim_wait_first_sample","no timestamp yet");
      SleepUntil(now+setup.period_s);
      continue;
    }
    else if(!Healthy(feedback,now,error)) reason="startup_"+error;
    else { startup.Update(feedback.sample,now); if(startup.fault()) reason=startup.reason(); }
    ui.status.reason=reason.empty() ? startup.reason() : reason;
    Snapshot(ui.phase,ui.status.reason); // 在任何拒绝/异常之前保存实际快照。
    ui.Emit(now);
    if(!reason.empty()) throw std::runtime_error(reason);
    if(startup.reason()=="startup_settled") break;
    SleepUntil(now+setup.period_s);
  } while(true);
  ac::Controller controller(policy);
  const double move_timeout_s=15;
  // 使用SDK已经发送的目标初始化，避免用重力下垂后的反馈重设目标。
  if (!controller.Initialize(feedback.sample,Now(),backend.LastTarget(),startup.feedback_anchor())) {
    Event("initialization_failed",controller.status().reason);
    throw std::runtime_error(controller.status().reason);
  }
  Event("initialized","SDK target retained; fixed session feedback offset; position stability window");
  auto baseline=Open("feedback_baseline.json");
  baseline << "{\"command_origin\":" << Array(controller.origin())
    << ",\"feedback_anchor\":" << Array(startup.feedback_anchor())
    << ",\"feedback_offset\":" << Array(controller.feedback_offset())
    << ",\"zero_calibrated\":false,\"scope\":\"session_checks_only\"}\n"; baseline.close();
  ui.feedback=feedback; ui.status=controller.status();
  ui.command_origin=controller.origin(); ui.feedback_anchor=feedback.sample.q; ui.has_anchor=true;
  ui.submitted=controller.status(); ui.has_ref=true;
  ui.phase="initialized"; ui.Ready("initialized");
  double last_heartbeat=Now();
  LoopPlan loop;
  std::array<Record,5> records{};
  auto SaveRecords=[&] {
    auto file=Open("direction_observations.json");
    file << "{\"schema\":\"X4_ARM_DIRECTION_OBSERVATIONS/1\",\"backend\":" << Quote(o.backend)
      << ",\"mpc_mapping_calibrated\":false,\"zero_calibrated\":false,\"observations\":[";
    bool first=true;
    for(int j=0;j<5;++j) if(records[j].set) {
      if(!first) file << ',';
      first=false;
      file << "{\"command_joint\":" << j+1 << ",\"observed_physical_joint\":" << records[j].physical
        << ",\"direction\":" << Quote(records[j].direction) << ",\"note\":" << Quote(records[j].note) << '}';
    }
    file << "]}\n";
  };
  SaveRecords();
  int selected=0, demo_moves=0, exit_code=0;
  bool quit=false, positive_observation=false, was_settled=false, session_stopped=false;
  std::string pending, termination="quit";
  bool eof=false;
  double tick=Now(), last_progress=tick;
  ac::Output last_sent=controller.status();
  bool sent=false;
  uint64_t observed_move=0;
  std::array<bool,5> deviation_warned{};
  std::cout << "初始化完成，等待位置稳定。输入status查看五轴；输入help查看命令。\n" << std::flush;
  while(!quit) {
    feedback=backend.Read(); const double now=Now();
    const bool healthy=Healthy(feedback,now,error);
    std::vector<std::string> lines;
    if(!o.demo) {
      for(const auto& candidate:InputBatch(pending,eof)) {
        std::istringstream check(candidate); std::string cmd,extra; check>>cmd;
        if(o.ui && cmd=="heartbeat" && !(check>>extra)) last_heartbeat=now;
        else lines.push_back(candidate);
      }
    }
    auto output=controller.Tick(feedback.sample,now);
    ui.feedback=feedback; ui.status=output;
    const bool lease_expired=o.ui && now-last_heartbeat>.5;
    if (!healthy || output.fault || interrupted || lease_expired) {
      // 即使故障由外层CAN状态触发，也仅冻结最后成功提交的位置，不采纳坏反馈。
      output=sent ? last_sent : controller.status(); output.dq.fill(0);
      output.initialized=true;
      std::string write_error;
      if(backend.Write(output,&write_error)) {
        stop_guard.last=output; stop_guard.sent=true;
        ui.submitted=output; ui.has_ref=true;
      }
      termination=interrupted ? "signal" : lease_expired ? "ui_heartbeat_timeout" : healthy ? controller.status().reason : error;
      Event("halt",termination); exit_code=interrupted ? 130 : 3;
      break;
    }
    if(!backend.Write(output,&error)) { Event("write_rejected",error); termination=error; exit_code=3; break; }
    if(!sent) Event("sdk_target_channel_enabled","validated target seeded before SetEnabled(true)");
    last_sent=output; sent=true;
    stop_guard.last=output; stop_guard.sent=true;
    ui.submitted=output; ui.has_ref=true;
    if(feedback.snapshot_attempts>1)
      Event("snapshot_retry",std::to_string(feedback.snapshot_attempts)+" attempts; original timestamps retained");
    if(policy.direction_only) for(int j=0;j<5;++j) {
      const double residual=feedback.sample.q[j]-controller.feedback_offset()[j]-output.q[j];
      const bool warn=(!output.moving || j!=output.active_joint) && std::abs(residual)>policy.max_other_motion_rad;
      if(warn && !deviation_warned[j]) {
        Event("angular_deviation_ignored","J"+std::to_string(j+1)+" "+std::to_string(residual/ac::kDeg)+" deg");
        std::cout << "方向模式：J" << j+1 << "偏移" << residual/ac::kDeg << "°，仅记录，继续观察方向\n" << std::flush;
      }
      deviation_warned[j]=warn;
    }
    if(output.completed_move_id!=observed_move) {
      observed_move=output.completed_move_id;
      observations << "{\"t\":" << now << ",\"id\":" << observed_move
        << ",\"joint\":" << output.completed_joint+1
        << ",\"command_delta\":" << output.completed_command_delta
        << ",\"feedback_start\":" << Array(output.feedback_start)
        << ",\"feedback_end\":" << Array(output.feedback_end)
        << ",\"feedback_delta\":" << Array(output.feedback_delta)
        << ",\"response_threshold_rad\":" << output.response_threshold_rad
        << ",\"response_direction\":" << Quote(output.response_direction) << "}\n" << std::flush;
      Event("motion_observed","J"+std::to_string(output.completed_joint+1)+" "+output.response_direction);
      std::cout << "方向结果 J" << output.completed_joint+1 << ": " << output.response_direction
                << "，实际变化 " << output.feedback_delta[output.completed_joint]/ac::kDeg << "°\n" << std::flush;
      if(policy.direction_only && output.response_direction=="no_response") positive_observation=false;
      if(policy.direction_only && loop.active && output.response_direction!="same") {
        loop.Clear(); positive_observation=false;
        Event("loop_cancelled",output.response_direction+"; no further motion queued");
        std::cout << "方向未确认，已取消自动循环；会话保持，可用status查看。\n" << std::flush;
      }
    }
    csv << now << ',' << selected+1 << ',' << output.moving << ',' << output.settled << ',' << output.fault
      << ',' << feedback.mode_state_synthetic;
    for(int j=0;j<5;++j) csv << ',' << feedback.sample.q[j] << ',' << feedback.sample.dq[j]
      << ',' << output.q[j] << ',' << output.dq[j] << ',' << feedback.sample.stamp_s[j]
      << ',' << feedback.sample.fault_bits[j] << ',' << feedback.sample.mode_state[j] << ',' << feedback.raw_mode_state[j]
      << ',' << feedback.torque[j] << ',' << feedback.temperature[j];
    csv << '\n';
    if(output.settled && !was_settled && !session_stopped) {
      std::cout << "已稳定，可操作 J" << selected+1 << '\n' << std::flush;
      Event("settled","J"+std::to_string(selected+1)); last_progress=now;
    }
    was_settled=output.settled;
    std::string line;
    if(o.demo) {
      if(output.settled) {
        if(policy.direction_only && observed_move && output.response_direction!="same") {
          quit=true; termination="direction_unconfirmed";
        }
        else if(demo_moves==10) { quit=true; termination="demo_complete"; }
        else {
          selected=demo_moves/2;
          if(!controller.StartMove(selected,(demo_moves%2==0 ? 1 : -1)*policy.step_rad,feedback.sample,now))
            throw std::runtime_error("demo move rejected: "+controller.status().reason);
          Event("demo_move","J"+std::to_string(selected+1)+(demo_moves%2==0 ? "+" : "-"));
          ++demo_moves; last_progress=now;
        }
      } else if(now-last_progress>move_timeout_s) { termination="settle_timeout"; exit_code=3; break; }
    } else {
      // 心跳已独立处理；quit > stop > pause，多动作不积压。
      for(const auto& candidate:lines) {
        std::istringstream check(candidate); std::string first; check >> first;
        if(first=="quit") { line="quit"; break; }
        if(first=="stop") line="stop";
        else if(first=="pause" && line.empty()) line="pause";
      }
      if(line.empty() && lines.size()==1) line=lines.front();
      else if(line.empty() && lines.size()>1) {
        Event("batch_rejected",std::to_string(lines.size())+" commands; no queued motion");
        std::cout << "拒绝整批命令：每次输入一条，动作不会排队执行\n" << std::flush;
      }
      if(!line.empty()) {
      Event("input",line);
      std::istringstream input(line); std::string command,extra; input >> command;
      if(command=="quit") { quit=true; termination="quit"; }
      else if(command=="status") Status(controller,feedback.sample,selected);
      else if(command=="help") Help();
      else if(command=="stop") {
        controller.Stop(now); positive_observation=false; session_stopped=true; loop.Clear();
        Event("software_stop","freeze last emitted target; session movement ended");
        std::cout << "本轮点动已终止并保持目标；可用status/help/quit\n";
      }
      else if(session_stopped) std::cout << "本轮已停止，只允许status/help/quit\n";
      else if(command=="pause") {
        loop.pause=true; Event("loop_pause_requested","finish current round trip");
        std::cout << "将完成当前往返并在原点暂停\n";
      }
      else if(command=="loop") {
        int count=0;
        if(!(input>>count) || (input>>extra) || count<1 || count>10)
          std::cout << "loop次数须为1..10\n";
        else if(loop.active || !output.settled || !AtOrigin(controller))
          std::cout << "循环需先回原点并等待稳定，且不能重叠启动\n";
        else {
          loop.active=true; loop.pause=false; loop.remaining=count;
          loop.stage=LoopPlan::Stage::ready; loop.since=now;
          positive_observation=false; Event("loop_started",std::to_string(count));
        }
      }
      else if(loop.active && command!="record") std::cout << "循环中只允许pause/stop/quit/status/help或端点record\n";
      else if(command=="+" || command=="-") {
        if(input >> extra) std::cout << "拒绝：命令不接受额外参数\n";
        else if(controller.StartMove(selected,(command=="+" ? 1 : -1)*policy.step_rad,feedback.sample,now)) {
          positive_observation=command=="+"; Event("move","J"+std::to_string(selected+1)+command);
        } else { std::cout << "拒绝：" << controller.status().reason << '\n'; Event("rejected",controller.status().reason); }
      } else if(command=="next" || command=="select") {
        int next=selected+2;
        const bool parsed=command=="next" || bool(input >> next);
        if(!parsed || (input >> extra) || next<1 || next>5) std::cout << "关节编号为1..5\n";
        else if(!output.settled || !AtOrigin(controller)) std::cout << "先反向回到本轮初始目标并等待稳定，再切轴\n";
        else { selected=next-1; positive_observation=false; Status(controller,feedback.sample,selected); }
      } else if(command=="record") {
        int physical=0; std::string direction,note;
        input >> physical >> direction; std::getline(input,note);
        bool duplicate=false;
        for(int j=0;j<5;++j) if(j!=selected && records[j].set && records[j].physical==physical) duplicate=true;
        if(!output.settled || !positive_observation) std::cout << "先执行+动作并等待稳定，再记录正方向观察\n";
        else if(physical<1 || physical>5 || (direction!="same" && direction!="opposite") || duplicate)
          std::cout << "拒绝：物理关节1..5，方向same/opposite，不能重复映射到其他已确认轴\n";
        else { records[selected]={true,physical,direction,note}; SaveRecords(); Event("observation_saved",line); }
      } else if(!command.empty()) std::cout << "未知命令，输入help查看\n";
      std::cout << std::flush;
      } else if(eof) { quit=true; termination="stdin_eof"; }
    }
    if(loop.active && !quit && !session_stopped) {
      auto Move=[&](int sign) {
        if(!controller.StartMove(selected,sign*policy.step_rad,feedback.sample,now))
          throw std::runtime_error("loop move rejected: "+controller.status().reason);
        positive_observation=sign>0; loop.since=now;
        Event("loop_move","J"+std::to_string(selected+1)+(sign>0 ? "+" : "-"));
      };
      switch(loop.stage) {
        case LoopPlan::Stage::ready:
          if(loop.pause) { loop.Clear(); Event("loop_complete","paused at origin"); }
          else { Move(1); loop.stage=LoopPlan::Stage::positive; }
          break;
        case LoopPlan::Stage::positive:
          if(output.settled) { loop.stage=LoopPlan::Stage::positive_dwell; loop.since=now; }
          break;
        case LoopPlan::Stage::positive_dwell:
          if(output.settled && now-loop.since>=.5) { Move(-1); loop.stage=LoopPlan::Stage::returning; }
          break;
        case LoopPlan::Stage::returning:
          if(output.settled) { loop.stage=LoopPlan::Stage::origin_dwell; loop.since=now; }
          break;
        case LoopPlan::Stage::origin_dwell:
          if(output.settled && now-loop.since>=.5) {
            --loop.remaining;
            if(loop.pause || loop.remaining==0) { loop.Clear(); Event("loop_complete","returned to origin"); }
            else { Move(1); loop.stage=LoopPlan::Stage::positive; }
          }
          break;
      }
      if(loop.active && now-loop.since>move_timeout_s) { termination="loop_settle_timeout"; exit_code=3; break; }
    }
    ui.status=controller.status(); ui.selected=selected; ui.session_stopped=session_stopped;
    ui.loop_active=loop.active; ui.loop_remaining=loop.remaining; ui.Emit(now);
    // 防止跟不上时补发一串历史点；下一拍由状态机检测超时并冻结目标。
    tick+=setup.period_s; if(tick<Now()) tick=Now();
    SleepUntil(tick);
  }
  stop_guard.Close();
  ui.Terminal(termination,exit_code!=0);
  Event("stopped",termination+"; SDK Stop is not disable");
  SaveRecords();
  auto result=Open("result.json");
  result << "{\"exit_code\":" << exit_code << ",\"reason\":" << Quote(termination)
    << ",\"completed_demo_moves\":" << demo_moves << ",\"hardware_alignment_validated\":false}\n";
  std::cout << "结束：" << termination << "；记录：" << o.output << "\n";
  return exit_code;
  } catch(const std::exception& e) {
    // 捕获后先保留故障前快照；Stop会改变SDK状态，不能覆盖诊断现场。
    const std::string reason=e.what();
    try { Snapshot(ui.phase,reason); Event("failed",reason); events.flush(); } catch(...) {}
    stop_guard.Close();
    ui.Terminal(reason,true);
    auto result=Open("result.json");
    result << "{\"exit_code\":2,\"reason\":" << Quote(reason)
      << ",\"phase\":" << Quote(ui.phase) << ",\"feedback_observed\":" << (ui.feedback_observed ? "true":"false")
      << ",\"hardware_alignment_validated\":false}\n";
    throw;
  }
}
}  // namespace
int main(int argc,char** argv) {
  try {
    if(argc==1 || (argc==2 && std::string(argv[1])=="--help")) { Help(); return 0; }
    const auto options=Parse(argc,argv);
    struct sigaction action{};
    action.sa_handler=Signal; sigemptyset(&action.sa_mask);
    // 不使用SA_RESTART，使终端确认/输入能在取消时返回。
    for(int sig:{SIGINT,SIGTERM,SIGHUP,SIGPIPE})
      if(sigaction(sig,&action,nullptr)!=0) throw std::runtime_error("sigaction failed");
    return Run(options);
  } catch(const std::exception& e) { std::cerr << "拒绝/失败：" << e.what() << '\n'; return 2; }
}
