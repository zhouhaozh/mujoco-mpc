#pragma once
// 可移植命令日志；来源/采样时间缺失必须显式保留，不能补造传感器时间。
#include "runtime/robot_io.h"
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace sckd::io {
using Json = nlohmann::json;
inline const char* ClockName(Clock c) { return c == Clock::kSimulation ? "simulation" : "monotonic"; }
inline const char* SourceName(Source s) {
  switch (s) {
    case Source::kSimulationTruth: return "simulation_truth";
    case Source::kMeasured: return "measured";
    case Source::kEstimated: return "estimated";
    case Source::kCommandIntegrated: return "command_integrated";
    default: return "unavailable";
  }
}
inline Json TimeJson(Time t) { return {{"seconds",t.seconds},{"clock",ClockName(t.clock)}}; }
inline Time TimeFromJson(const Json& j) {
  const auto c=j.at("clock").get<std::string>();
  if(c!="simulation" && c!="monotonic") throw std::runtime_error("unknown clock: "+c);
  const double s=j.at("seconds").get<double>();
  if(!std::isfinite(s)) throw std::runtime_error("nonfinite timestamp");
  return {s,c=="simulation"?Clock::kSimulation:Clock::kMonotonic};
}
inline Json ToJson(const Command& c) {
  return {{"sequence",c.sequence},{"plan_id",c.plan_id},{"issued",TimeJson(c.issued)},
    {"expires",TimeJson(c.expires)},{"frame_id",c.frame_id},{"body_twist",c.body_twist},
    {"q",c.q},{"dq",c.dq},{"kp",c.kp},{"kd",c.kd},{"tau_ff",c.tau_ff}};
}
template<std::size_t N> inline std::array<double,N> ArrayFromJson(const Json& j) {
  if(!j.is_array() || j.size()!=N) throw std::runtime_error("wrong array dimension");
  std::array<double,N> out{};
  for(std::size_t i=0;i<N;++i) {
    out[i]=j.at(i).get<double>();
    if(!std::isfinite(out[i])) throw std::runtime_error("nonfinite array element");
  }
  return out;
}
inline std::uint64_t SequenceFromJson(const Json& j) {
  if(!j.is_number_unsigned()) throw std::runtime_error("sequence must be unsigned integer");
  return j.get<std::uint64_t>();
}
inline Command CommandFromJson(const Json& j) {
  Command c;
  c.sequence=SequenceFromJson(j.at("sequence")); c.plan_id=SequenceFromJson(j.at("plan_id"));
  c.issued=TimeFromJson(j.at("issued")); c.expires=TimeFromJson(j.at("expires"));
  c.frame_id=j.at("frame_id").get<std::string>();
  c.body_twist=ArrayFromJson<3>(j.at("body_twist"));
  c.q=ArrayFromJson<5>(j.at("q")); c.dq=ArrayFromJson<5>(j.at("dq"));
  c.kp=ArrayFromJson<5>(j.at("kp")); c.kd=ArrayFromJson<5>(j.at("kd"));
  c.tau_ff=ArrayFromJson<5>(j.at("tau_ff"));
  const auto v=Validate(c,c.issued,{true,false,c.frame_id});
  if(!v.submitted) throw std::runtime_error("invalid command: "+v.reason);
  return c;
}
inline Json ObservationToJson(const Observation& o) {
  Json joints=Json::array();
  for(const auto& j:o.joints) joints.push_back({{"q",j.q},{"dq",j.dq},{"torque",j.torque},
    {"temperature",j.temperature},{"available",j.available},{"stamp_known",j.stamp_known},
    {"stamp",j.stamp_known?TimeJson(j.stamp):Json(nullptr)}, {"fault_bits",j.fault_bits},
    {"position_source",SourceName(j.position_source)},{"velocity_source",SourceName(j.velocity_source)}});
  return {{"sequence",o.sequence},{"received",TimeJson(o.received)},{"frame_id",o.frame_id},
    {"base_pose",o.base_pose},{"body_twist",o.body_twist},{"base_available",o.base_available},
    {"base_stamp_known",o.base_stamp_known},{"base_stamp",o.base_stamp_known?TimeJson(o.base_stamp):Json(nullptr)},
    {"base_source",SourceName(o.base_source)},{"joints",joints},{"ready",o.ready},
    {"arm_degraded",o.arm_degraded},{"manual_takeover",o.manual_takeover},
    {"coherent_snapshot",o.coherent_snapshot}};
}
class CommandLog {
 public:
  CommandLog(const std::string& path,Json metadata) {
    const auto parent=std::filesystem::path(path).parent_path();
    if(!parent.empty()) std::filesystem::create_directories(parent);
    file_=std::fopen(path.c_str(),"wx");
    if(!file_) throw std::runtime_error("cannot exclusively create log "+path+": "+std::strerror(errno));
    try { Line({{"schema","X4_ROBOT_IO/1"},{"type","manifest"},{"metadata",std::move(metadata)}}); }
    catch(...) {std::fclose(file_);file_=nullptr;throw;}
  }
  ~CommandLog() { if(file_) std::fclose(file_); }
  CommandLog(const CommandLog&)=delete;
  CommandLog& operator=(const CommandLog&)=delete;
  void Write(const Command& c,const Observation& o,const WriteResult& r) {
    Line({{"schema","X4_ROBOT_IO/1"},{"type","sample"},{"command",ToJson(c)},
      {"observation",ObservationToJson(o)},
      {"result",{{"submitted",r.submitted},{"reason",r.reason}}}});
  }
 private:
  void Line(const Json& value) {
    const auto line=value.dump()+"\n";
    if(std::fwrite(line.data(),1,line.size(),file_)!=line.size() || std::fflush(file_)!=0)
      throw std::runtime_error("robot IO log write failed");
  }
  std::FILE* file_=nullptr;
};
struct RecordedCommands { Json metadata; std::vector<Command> commands; };
inline RecordedCommands ReadCommands(const std::string& path) {
  std::ifstream f(path); if(!f) throw std::runtime_error("cannot open command log: "+path);
  RecordedCommands out; std::string line; std::size_t number=0;
  while(std::getline(f,line)) {
    ++number;
    if(line.empty()) throw std::runtime_error("empty log row "+std::to_string(number));
    const auto j=Json::parse(line);
    if(j.at("schema")!="X4_ROBOT_IO/1") throw std::runtime_error("unsupported IO schema");
    if(number==1) {
      if(j.at("type")!="manifest") throw std::runtime_error("missing manifest");
      out.metadata=j.at("metadata"); continue;
    }
    if(j.at("type")!="sample") throw std::runtime_error("unexpected log record");
    if(!j.at("result").at("submitted").get<bool>()) throw std::runtime_error("log contains rejected command");
    auto c=CommandFromJson(j.at("command"));
    if(!out.commands.empty()) {
      const auto& prev=out.commands.back();
      if(c.sequence<=prev.sequence || c.issued.clock!=prev.issued.clock ||
         c.issued.seconds<=prev.issued.seconds || c.frame_id!=prev.frame_id)
        throw std::runtime_error("nonmonotonic or mixed-frame command stream");
    }
    out.commands.push_back(std::move(c));
  }
  if(!f.eof() || out.commands.empty()) throw std::runtime_error("empty or unreadable command stream");
  return out;
}
}  // namespace sckd::io
