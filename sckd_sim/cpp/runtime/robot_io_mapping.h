#pragma once
#include "runtime/robot_io_log.h"

namespace sckd::io {
// 仅做具名模型关节系之间的映射；SDK内部电机sign/offset不属于本层。
struct JointFrameMap {
  std::string from_frame, to_frame, version;
  std::array<double,5> sign{}, offset_rad{};
  bool calibrated=false;
  static JointFrameMap Load(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot read joint frame map");
    Json j; f>>j;
    if(j.at("schema")!="X4_JOINT_FRAME_MAP/1") throw std::runtime_error("unknown frame-map schema");
    JointFrameMap m;
    m.from_frame=j.at("from_frame");m.to_frame=j.at("to_frame");m.version=j.at("version");
    m.sign=ArrayFromJson<5>(j.at("sign"));m.offset_rad=ArrayFromJson<5>(j.at("offset_rad"));
    m.calibrated=j.at("calibrated").get<bool>();
    if(m.from_frame.empty()||m.to_frame.empty()||m.version.empty()) throw std::runtime_error("missing map identity");
    for(double s:m.sign) if(s!=1 && s!=-1) throw std::runtime_error("map sign must be +1/-1");
    return m;
  }
  Command Apply(Command c) const {
    if(c.frame_id!=from_frame) throw std::runtime_error("frame-map input mismatch");
    for(int j=0;j<5;++j) {c.q[j]=sign[j]*c.q[j]+offset_rad[j];c.dq[j]*=sign[j];c.tau_ff[j]*=sign[j];}
    c.frame_id=to_frame;return c;
  }
};
}  // namespace sckd::io
