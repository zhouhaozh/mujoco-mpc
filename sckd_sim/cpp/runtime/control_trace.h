#pragma once
// 日志资源由运行域拥有；不使用全局FILE，不改变既有事件字段。
#include <cstdio>
#include <optional>
#include <string>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "runtime/contracts.h"
namespace sckd::runtime {
inline nlohmann::json Vec(const double* p,int n){return std::vector<double>(p,p+n);}
class ControlTrace {
  FILE* file_=nullptr;
  std::optional<std::string> path_;
  std::string track_id_;
  Sequence observation_seq_=0,state_seq_=0;
  std::optional<double> last_capture_,last_receive_;
 public:
  explicit ControlTrace(const std::optional<std::string>& path):path_(path){if(path){file_=std::fopen(path->c_str(),"wx");if(!file_)throw std::runtime_error("cannot create control diagnostic file");}}
  ~ControlTrace(){if(file_)std::fclose(file_);}
  ControlTrace(const ControlTrace&)=delete;ControlTrace& operator=(const ControlTrace&)=delete;
  bool enabled()const{return file_!=nullptr;}
  void RecordConfig(const nlohmann::json& config){
    if(!path_)return;
    FILE* f=std::fopen((*path_+".config.json").c_str(),"wx");
    if(!f)throw std::runtime_error("cannot create resolved runtime configuration");
    std::fprintf(f,"%s\n",config.dump(2).c_str());std::fclose(f);
  }
  void SetTrial(int trial){track_id_="trial-"+std::to_string(trial);observation_seq_=state_seq_=0;last_capture_.reset();last_receive_.reset();}
  void SetInput(const BallObservation& ball,const RobotEstimate& robot,bool have_ball){
    state_seq_=robot.state_seq;
    if(have_ball){observation_seq_=ball.identity.observation_seq;last_capture_=ball.identity.captured.seconds;last_receive_=ball.identity.received.seconds;}
  }
  void Emit(nlohmann::json row){if(file_){
    row["schema"]="SCKD_CONTROL_TRACE/1";
    row["runtime_identity"]={{"clock_id","simulation"},{"track_id",track_id_},{"observation_seq",observation_seq_},{"state_seq",state_seq_},
      {"last_capture_s",last_capture_?nlohmann::json(*last_capture_):nlohmann::json(nullptr)},
      {"last_receive_s",last_receive_?nlohmann::json(*last_receive_):nlohmann::json(nullptr)}};
    std::fprintf(file_,"%s\n",row.dump().c_str());
  }}
};
}
