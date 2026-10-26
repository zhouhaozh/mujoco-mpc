// 命令日志预检/仿真回放。无硬件模式，不调用 Robot::Init。
#include "runtime/robot_io_log.h"
#include "runtime/robot_io_mapping.h"
#include "runtime/mujoco_robot_backend.h"
#include "hal/sim_backend.h"
#include "control/chassis_servo.h"
#ifdef SCKD_WITH_X4_SDK
#include "runtime/x4_sdk_backend.h"
#endif
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

using namespace sckd::io;
int main(int argc,char** argv) {
 try {
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2) {
    if(i+1>=argc || std::string(argv[i]).rfind("--",0)!=0) throw std::runtime_error("expected --key value");
    if(!args.emplace(argv[i],argv[i+1]).second) throw std::runtime_error("duplicate argument");
  }
  for(const auto& a:args) if(a.first!="--input"&&a.first!="--output"&&a.first!="--backend"&&
      a.first!="--model-root"&&a.first!="--sdk-config"&&a.first!="--sdk-log-dir"&&a.first!="--joint-map")
    throw std::runtime_error("unknown argument "+a.first);
  const auto required=[&](const char* key)->std::string {
    if(!args.count(key)||args.at(key).empty()) throw std::runtime_error(std::string("missing ")+key);
    return args.at(key);
  };
  const std::string kind=required("--backend");
  if(kind!="validate"&&kind!="mujoco"&&kind!="sdk-sim") throw std::runtime_error("backend must be validate, mujoco or sdk-sim; hardware is unavailable");
  auto recorded=ReadCommands(required("--input"));
  Json metadata=recorded.metadata;
  metadata["replay_source"]=std::filesystem::absolute(required("--input")).string();
  metadata["replay_backend"]=kind;
  if(args.count("--joint-map")) {
    if(kind!="sdk-sim") throw std::runtime_error("joint mapping is only supported for sdk-sim replay");
    const auto map=JointFrameMap::Load(args.at("--joint-map"));
    for(auto& c:recorded.commands)c=map.Apply(c);
    metadata["joint_map"]={{"path",std::filesystem::absolute(args.at("--joint-map")).string()},
      {"version",map.version},{"calibrated",map.calibrated},{"from_frame",map.from_frame},
      {"to_frame",map.to_frame},{"sign",map.sign},{"offset_rad",map.offset_rad}};
  }
  const bool sdk=kind=="sdk-sim";
  const Capabilities caps{!sdk,false,sdk?"sdk_model":kind=="mujoco"?"mpc_model":recorded.commands.front().frame_id};
  if(kind=="mujoco" && metadata.contains("resolved_config") &&
      metadata.at("resolved_config").at("derived").value("measured_replay",false))
    throw std::runtime_error("MuJoCo replay does not support measured_replay model collision overrides");
#ifdef SCKD_WITH_X4_SDK
  X4SdkJointEncoding sdk_encoding;
  if(sdk) {
    std::string error;
    if(!LoadX4SdkJointEncoding(required("--sdk-config"),&sdk_encoding,&error))
      throw std::runtime_error("SDK encoding config: "+error);
  }
#endif
  // 全文件预检在任何仿真工厂启动之前；非零前馈不会被自动删除。
  for(const auto& c:recorded.commands) {
    if(kind=="mujoco"&&c.issued.clock!=Clock::kSimulation) throw std::runtime_error("MuJoCo replay requires simulation timestamps");
    const auto result=Validate(c,c.issued,caps);
    if(!result.submitted) throw std::runtime_error("preflight: "+result.reason);
    if(sdk) {
#ifdef SCKD_WITH_X4_SDK
      auto probe=c;
      probe.issued.clock=probe.expires.clock=Clock::kMonotonic;
      const auto checked=ValidateX4SdkCommand(probe,probe.issued,caps,sdk_encoding);
      if(!checked.submitted)throw std::runtime_error("SDK preflight: "+checked.reason);
#else
      throw std::runtime_error("built without X4 SDK; configure X4_SDK_ROOT");
#endif
    }
  }
  if(kind=="validate") {
    std::cout<<Json({{"schema","X4_ROBOT_IO_VALIDATION/1"},{"samples",recorded.commands.size()},
      {"first_time",TimeJson(recorded.commands.front().issued)},
      {"last_time",TimeJson(recorded.commands.back().issued)},{"hardware_access",false}}).dump(2)<<'\n';
    return 0;
  }
  // 提前检查，最终仍由CommandLog独占创建以防覆盖。
  if(std::filesystem::exists(required("--output"))) throw std::runtime_error("output already exists");
  metadata["hardware_access"]=false;
  metadata["replay_command_frame"]=caps.frame_id;
  std::unique_ptr<sckd::SimBackend> sim;
  std::unique_ptr<sckd::ChassisServo> servo;
  std::unique_ptr<RobotBackend> backend;
  double origin=0;
  if(kind=="mujoco") {
    const auto root=std::filesystem::absolute(required("--model-root"));
    const auto path=[&](const char* key){return (root/recorded.metadata.at(key).get<std::string>()).string();};
    sim=std::make_unique<sckd::SimBackend>(path("scene_path"),path("motors_path"),
      path("robot_variant_path"),path("chassis_path"));
    if(!sim->ok()||!sim->reset(0)) throw std::runtime_error("MuJoCo initialization failed");
    if(std::abs(sim->model()->opt.timestep-.001)>1e-10 ||
       std::abs(metadata.at("physics_step_s").get<double>()-.001)>1e-10)
      throw std::runtime_error("MuJoCo command replay requires 1 ms physical steps");
    sim->enable(-1,true);
    const auto& hold=metadata.at("initial_motor_hold");
    const double hold_kp=hold.at("kp").get<double>(),hold_kd=hold.at("kd").get<double>();
    if(!std::isfinite(hold_kp)||!std::isfinite(hold_kd)||hold_kp<0||hold_kd<0)
      throw std::runtime_error("invalid initial hold gains");
    sim->hold_arm(hold_kp,hold_kd);
    if(recorded.metadata.contains("mujoco_initial_state")) {
      const auto& initial=recorded.metadata.at("mujoco_initial_state");
      auto* d=sim->data_mut(); const auto* m=sim->model();
      const auto restore=[&](const char* key,mjtNum* target,int n) {
        const auto& values=initial.at(key);
        if(!values.is_array()||values.size()!=static_cast<std::size_t>(n)) throw std::runtime_error("initial-state shape mismatch");
        for(int i=0;i<n;++i) {target[i]=values.at(i).get<double>();if(!std::isfinite(target[i]))throw std::runtime_error("nonfinite initial state");}
      };
      restore("qpos",d->qpos,m->nq);restore("qvel",d->qvel,m->nv);restore("act",d->act,m->na);
      d->time=initial.at("time").get<double>();
      if(!std::isfinite(d->time)||d->time<0)throw std::runtime_error("invalid initial time");
      if(std::abs(recorded.commands.front().issued.seconds-d->time-.001)>1e-8)
        throw std::runtime_error("first command does not follow recorded initial state by 1 ms");
      mj_forward(m,d);
    } else throw std::runtime_error("missing recorded initial state");
    servo=std::make_unique<sckd::ChassisServo>(sim->chassis_config(),sim.get());
    // 入口的底盘限率模式也是执行模型的一部分，回放必须继承。
    servo->set_steering_rate_limits(metadata.value("direction_limited",true),metadata.value("feedforward_limited",true));
    backend=std::make_unique<MujocoRobotBackend>(*sim,*servo);
    metadata["replay_model_root"]=root.string();
    metadata["replay_scope"]="actuator commands and initial state; external ball forces and collision overrides are not replayed";
    for(std::size_t i=1;i<recorded.commands.size();++i)
      if(std::abs(recorded.commands[i].issued.seconds-recorded.commands[i-1].issued.seconds-.001)>1e-8)
        throw std::runtime_error("MuJoCo replay currently requires recorded 1 ms commands");
  } else {
#ifdef SCKD_WITH_X4_SDK
    std::string error;
    backend=CreateX4SdkSimulation(required("--sdk-config"),required("--sdk-log-dir"),&error);
    if(!backend)throw std::runtime_error("SDK simulation: "+error);
    origin=X4SdkMonotonicNow().seconds+.03-recorded.commands.front().issued.seconds;
    metadata["clock_rebase_offset_s"]=origin;
    metadata["sdk_config"]=std::filesystem::absolute(args.at("--sdk-config")).string();
#else
    throw std::runtime_error("built without X4 SDK; configure X4_SDK_ROOT");
#endif
  }
  CommandLog log(required("--output"),metadata);
  std::size_t count=0; double max_lateness=0;
  for(auto c:recorded.commands) {
    Time now=c.issued;
    if(sdk) {
#ifdef SCKD_WITH_X4_SDK
      c.issued={c.issued.seconds+origin,Clock::kMonotonic};
      c.expires={c.expires.seconds+origin,Clock::kMonotonic};
      now=X4SdkMonotonicNow();
      if(now.seconds<c.issued.seconds) std::this_thread::sleep_for(std::chrono::duration<double>(c.issued.seconds-now.seconds));
      now=X4SdkMonotonicNow();max_lateness=std::max(max_lateness,now.seconds-c.issued.seconds);
#endif
    }
    const auto result=backend->Write(c,now);
    if(!sdk && result.submitted)static_cast<MujocoRobotBackend*>(backend.get())->Tick(now);
    log.Write(c,backend->Read(now),result);
    if(!result.submitted)throw std::runtime_error("replay rejected at sequence "+std::to_string(c.sequence)+": "+result.reason);
    if(sim&&!sim->run(1))throw std::runtime_error("MuJoCo step failed");
    ++count;
  }
  std::cout<<Json({{"schema","X4_ROBOT_IO_REPLAY/1"},{"backend",kind},{"samples",count},
    {"max_lateness_s",max_lateness},{"hardware_access",false},
    {"note","command replay; no claim of physical equivalence or real-time acceptance"}}).dump(2)<<'\n';
  return 0;
 } catch(const std::exception& e) {std::cerr<<"robot_io_replay: "<<e.what()<<'\n';return 2;}
}
